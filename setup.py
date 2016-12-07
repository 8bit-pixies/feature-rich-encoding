from setuptools import setup, find_packages

config = {
    'description': 'Feature Rich Encodings for Text Features',
    'author': 'Chapman Siu',
    'url': 'https://github.com/chappers/feature-rich-encoding',
    'download_url': 'https://github.com/chappers/feature-rich-encoding',
    'author_email': 'chpmn.siu@gmail.com',
    'version': '0.1.0',
    'install_requires': [
        'gensim',
        'nltk',
        'scikit-learn'
      ],
    'packages': ['FeatureRichEncoding'],
    'name': 'FeatureRichEncoding',
    'include_package_data':True, 
    'zip_safe':False
}

setup(**config)


