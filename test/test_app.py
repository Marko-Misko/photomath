import os

import pytest

from src.backend import create_app


@pytest.fixture
def app():
    app = create_app()
    return app


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.mark.parametrize('image, expected', [
    ('test_0.jpg', 182),
    ('test_1.jpg', 2),
    ('test_2.jpg', 1)
])
def test_create_photo(client, image, expected):
    image_name = os.path.abspath(os.path.join(__file__, f'../images/{image}'))
    response = client.post('/', data={'image': (open(image_name, 'rb'), image_name)})
    assert bytes(f'Solution is {expected}', 'utf-8') in response.data
