# ml_library
Writing a foundational C++ library from scratch.

## Overview

This small project is for me to get familiar with the basic ideas of machine learning. Eventually, I want to use existing machine learning libraries with my deepened understanding of these basics to train a model and integrate it into an embedded system, such as an STM32 or ESP32. Alongside this, I will continue learning PCB design by building upon my first PCB-based project, `macroWAN`, where I will build a more complicated electrical circuit (this circuit is unknown as of writing this). Together, these two pieces will be used in a project involving all of these learned skills.

## Expected Timeline

1. Learn the basics of machine learning
2. a. Apply machine learning skills on a real dataset on a microcontroller

   b. Learn more complex PCB design
3. Create a more complex system combining machine learning, PCB design and embedded systems

## Part I: Autograd Engine

The first part of this project will be developing an Autograd engine. This is inspired from from Andrej Karpathy's micrograd engine, but written in C++. All the code will be inside `ml_library` with the header files inside `inc`, source files in `src` and any tests that I've written inside `tests`.

## Part 2: Neural Network

The second part of this project will be developing the neural network infrastructure to perform basic ML tasks.