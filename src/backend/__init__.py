import os

from flask import Flask

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}


def create_app(test_config=None):
    app = Flask(__name__, instance_relative_config=True)

    if test_config is None:
        app.config.from_pyfile('config.py', silent=True)
    else:
        app.config.from_mapping(test_config)

    os.makedirs(app.instance_path, exist_ok=True)

    from . import service
    app.register_blueprint(service.bp)
    app.add_url_rule('/', endpoint='service.index')

    return app
