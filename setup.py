import setuptools

with open("README.md", "r", encoding="utf-8") as file_handle:
    long_description = file_handle.read()

setuptools.setup(
    name='FiLM_image_encoder',
    version='0.0.1',
    author='Tyler Lum',
    author_email='tylergwlum@gmail.com',
    description='Feature-wise Linear Modulation (FiLM) integrated with image encoders',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/tylerlum/FiLM_image_encoder',
    project_urls = {
        'Bug Tracker': 'https://github.com/tylerlum/FiLM_image_encoder/issues',
    },
    license='MIT',
    packages=['FiLM_image_encoder'],
    install_requires=['torch'],
)
