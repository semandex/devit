"""
CLIP Evaluation

Requires path to csv containing [Query, GTP Label, Count]

For more information on metrics:
https://drive.google.com/file/d/1uG-0DVILDIYBrA-1ILXZGhWtpDNjyIi5/view?pli=1

Rank is how many results to look at, up to the length
of results returned for the query. Higher values will generally produce higher recall but lower precision.

TODO Normalized Discounted Cumulative Gain (nDCG)
TODO threshold

usage: clip_evaluation.py [-h] [-n DATASET_NAME] [-d DATASET_PATH]
                          [-u SEARCH_URL] [-o OUTPUT_FILENAME] [--no-print]

CLIP Evaluation

options:
  -h, --help            show this help message and exit
  -n DATASET_NAME, --dataset-name DATASET_NAME
                        Name of the VizX GT Dataset for filtering
  -d DATASET_PATH, --dataset-path DATASET_PATH
                        Path to VizX GT Dataset
  -u SEARCH_URL, --search-url SEARCH_URL
                        Base VizX Search URL
  -o OUTPUT_FILENAME, --output-filename OUTPUT_FILENAME
                        The name of the CSV to output. Default uses dataset
                        name.
  --no-print            Whether to disable printing of metric table.


To run with Docker, see README.md.
"""

import argparse
import functools
import os
from io import BytesIO
from itertools import accumulate
from time import sleep
from typing import List, Union

import PIL
import numpy as np
import pandas as pd
import requests
from PIL import Image
from tabulate import tabulate
from tqdm import tqdm
import cv2

GTP_COUNT_BUFFERS = [0, 10, 20]
RANKS = ['max_results', 5, 10, 25, 50, 100] + [f'GTP_count+{buffer}' for buffer in GTP_COUNT_BUFFERS]


class Evaluator:

    def __init__(self, dataset_name: str, dataset_path: str, search_url: str):
        self.name = dataset_name
        self.search_url = search_url

        df = pd.read_csv(dataset_path)
        df = df.dropna(subset=['GTP Label'])  # remove metadata rows
        columns_to_remove = [col for col in df.columns if 'count ' in col.lower() or 'of total' in col.lower()]
        df = df.drop(columns=columns_to_remove)

        self.df = df
        self.query_to_results = dict()

    @staticmethod
    def _create_image_new(image_bytes):
        try:
            pil_image = Image.open(BytesIO(image_bytes))
        except PIL.UnidentifiedImageError:
            cv2_image = cv2.imdecode(np.frombuffer(BytesIO(image_bytes).read(), np.uint8), cv2.IMREAD_UNCHANGED)
            if cv2_image is None:  # likely because it is AVIF
                raise ValueError
            cv2_image[:, :, [0, 1, 2]] = cv2_image[:, :, [2, 1, 0]]  # convert OpenCV BGR(A) to RGB(A)
            pil_image = Image.fromarray(cv2_image)
        return pil_image

    def _create_image(self, url, filepath):
        user_agent_headers = {  # prevent 403
            "User-Agent":
                "Mozilla/5.0 "
                "(Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.82 Safari/537.36"
        }
        get_fixed_params = functools.partial(requests.get, stream=True, headers=user_agent_headers)

        response = get_fixed_params(url)
        content = response.raw.read()

        try:
            if response.status_code != 200:
                raise ValueError(f'Status Code == {response.status_code}')
            image = self._create_image_new(content)
            image.save(filepath)
        except ValueError as e:
            message = ''
            if len(e.args) > 0:
                message = f': {e.args[0]}'

            raise ValueError(f'Error reading image at {url}{message}') from None

    def make_search_request(self, text: str = None, observations: List = (), page_size: int = 16,
                            min_score: float = None) -> dict:
        payload = {
            'category': 'tiles',
            'filters': [
                {
                    'id': 'lexicon',
                    'values': list(observations),
                    'mode': 'ALL_OF'
                }
            ],
            'sort': 'MOST_RELEVANT',
            'page': 1,
            'pageSize': page_size,
        }
        if text is not None:
            search_filter = {
                "id": "searchText",
                "text": text
            }
            if min_score is not None:  # otherwise default minScore will be used
                search_filter['minScore'] = min_score
            payload['filters'].append(search_filter)

        max_retries = 3
        retry = 0
        while True:
            try:
                response = requests.post(self.search_url, json=payload)
                assert response.status_code == 200
                return response.json()
            except requests.exceptions.RequestException as e:
                retry += 1
                if retry >= max_retries:
                    raise e from None
                print(f'POST request failed to {self.search_url} with payload {payload}: {e}')
                print('Retrying after 30s...')
                sleep(30)

    def get_all_search_results(self, min_score: float = None) -> None:
        pbar = tqdm(total=self.df.Counts.sum())

        for i, row in self.df.iterrows():
            query = row['Query']
            gtp_label = row['GTP Label']
            image_dir = f'data/images/{gtp_label[5:]}'
            os.makedirs(image_dir, exist_ok=True)

            try:
                search_result = self.make_search_request(observations=[gtp_label.lower()])
                gtp_count = search_result['total']
                assert gtp_count == row['Counts']

                search_result = self.make_search_request(observations=[gtp_label.lower()], page_size=gtp_count)
                results = [
                    {
                        'filename': f'{result["id"]}.png',
                        'url': result['document']['tileURL']
                    } for result in search_result['results']
                ]

                for result in results:
                    filepath = os.path.join(image_dir, result['filename'])
                    if not os.path.exists(filepath):
                        self._create_image(result['url'], filepath)

                    pbar.update()

            except requests.exceptions.RequestException:
                print(f'Could not resolve error for {query} ({gtp_label}). Skipping...')
                self.df.drop(i, inplace=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CLIP Evaluation')
    parser.add_argument('-n', '--dataset-name', type=str, required=True,
                        help='Name of the VizX GT Dataset for filtering')
    parser.add_argument('-d', '--dataset-path', type=str, required=True, help='Path to VizX GT Dataset')
    parser.add_argument('-u', '--search-url', type=str, help='Base VizX Search URL',
                        default='https://salsa1.semandex.net/v4s-beta/som/app/api/search')
    parser.add_argument('-s', '--min-score', type=float, help='The minScore to pass to VizX for searching.')
    parser.add_argument('-o', '--output-filename', type=str,
                        help='The name of the CSV to output. Default uses dataset name.')
    parser.add_argument('--no-print', action='store_true',
                        help='Whether to disable printing of metric table.')

    args = parser.parse_args()

    dataset_name = args.dataset_name
    dataset_path = args.dataset_path
    search_url = args.search_url
    output_filename = args.output_filename
    print(f'Processing dataset {dataset_name} at {dataset_path} using {search_url}')
    min_score = args.min_score
    if min_score is not None:
        print(f'Using {min_score:.3f} as minScore...')
        assert 0.0 <= min_score <= 1.0

    evaluator = Evaluator(dataset_name, dataset_path, search_url)
    evaluator.get_all_search_results(min_score=min_score)
    # df = evaluator.evaluate_dataset()
