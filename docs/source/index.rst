.. S3PRL documentation master file, created by
   sphinx-quickstart on Sun May 15 15:43:39 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

S3PRL
=====

.. image:: https://raw.githubusercontent.com/s3prl/s3prl/master/file/S3PRL-logo.png

**S3PRL** is a toolkit targeting for Self-Supervised Learning for speech processing.
Its full name is **S**\elf-**S**\upervised **S**\peech **P**\re-training and **R**\epresentation **L**\earning.
It supports the following three major features:

* **Pre-training**

   * You can train the following models from scratch:

   * *Mockingjay*, *Audio ALBERT*, *TERA*, *APC*, *VQ-APC*, *NPC*, and *DistilHuBERT*

* **Pre-trained models (Upstream) collection**

   * Easily load most of the existing upstream models with pretrained weights in a unified I/O interface.
   * Pretrained models are registered through torch.hub, which means you can use these models in your own project by one-line plug-and-play without depending on this toolkit's coding style.

* **Downstream Evaluation**

   * Utilize upstream models in lots of downstream tasks
   * The official implementation of the `SUPERB Benchmark <https://superbbenchmark.org/>`_


Getting Started
---------------

.. toctree::
   :caption: Getting started

   ./tutorial/installation.rst
   ./tutorial/upstream_collection.rst
   ./tutorial/problem.rst


How to Contribute
-----------------

.. toctree::
   :caption: How to Contribute

   ./contribute/public.rst
   ./contribute/private.rst


API Documentation
-----------------

.. autosummary::
   :caption: API Documentation
   :toctree: _autosummary
   :template: custom-module-template.rst
   :recursive:

   s3prl.nn
   s3prl.problem
   s3prl.task
   s3prl.dataio
   s3prl.metric
   s3prl.util


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
