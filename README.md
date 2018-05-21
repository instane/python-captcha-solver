# python-captcha-solver

This project created for solving captcha images from http://iq.karelia.ru 

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

What things you need to install the software and how to install them

First of all you need a python3 with pip and tesseract

```
# dnf install python3 python3-pip tesseract tesseract-devel
```

For next you will need pipenv

```
# pip3 install pipenv
```

### Installing

A step by step series of examples that tell you have to get a development env running

Copy repository to a local machine and cd into it

```
git clone https://github.com/instane/python-captcha-solver.git && cd python-captcha-solver
```

Install requirement libraries

```
pipenv install
```

Spawn a shell with the virtualenv activated

```
pipenv shell
```

Now you can run program

```
python pcs.py path-to-image.jpeg
```

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [OpenCV](https://opencv.org/) - Open Source Computer Vision Library
* [NumPy](http://www.numpy.org/) - fundamental package for scientific computing with Python
* [Tesseract](https://github.com/tesseract-ocr/tesseract) - Open Source OCR Engine

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/instane/python-captcha-solver/tags). 

## Authors

* **Evgeny** - *Initial work* - [instane](https://github.com/instane)
* **Ilya** - *Initial work* - [Klemushka](https://github.com/Klemushka)

See also the list of [contributors](https://github.com/python-captcha-solver/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone who's code was used
* Inspiration
* etc

