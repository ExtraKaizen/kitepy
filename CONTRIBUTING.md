# Contributing to kitepy 🚀

Thank you for your interest in contributing to **kitepy**! We welcome contributions from the community to help make deep learning "dead simple" for everyone.

## Table of Contents
1. [How Can I Contribute?](#how-can-i-contribute)
2. [Code of Conduct](#code-of-conduct)
3. [Setting Up Your Development Environment](#setting-up-your-development-environment)
4. [Creating a Pull Request](#creating-a-pull-request)
5. [Reporting Bugs](#reporting-bugs)
6. [Suggesting Features](#suggesting-features)

---

## How Can I Contribute?

There are many ways to help:
- **Code**: Implement new models (LLMs, VLMs), features, or fix bugs.
- **Documentation**: Improve existing docs or add new tutorials and examples.
- **Testing**: Add unit tests or integration tests to ensure stability.
- **Support**: Help others by answering questions in the GitHub Issues.

## Code of Conduct

Help us keep the `kitepy` community healthy and welcoming. Please be respectful and professional in all interactions.

## Setting Up Your Development Environment

1. **Fork the Repository**: Create your own copy of the `kitepy` repo.
2. **Clone the Repo**:
   ```bash
   git clone https://github.com/YourUsername/kitepy
   cd kitepy
   ```
3. **Install in Development Mode**:
   ```bash
   pip install -e .[all,dev]
   ```
4. **Create a Branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Creating a Pull Request

1. **Keep it focused**: One PR per feature or bug fix.
2. **Add Tests**: If you add code, please add corresponding tests in the `tests/` directory.
3. **Format Your Code**: We use `black` for formatting.
   ```bash
   black .
   ```
4. **Run Tests**:
   ```bash
   pytest tests/
   ```
5. **Submit**: Push your branch and open a PR on the `ExtraKaizen/kitepy` repository.

## Reporting Bugs

- Search existing issues to avoid duplicates.
- Provide a **Minimal Reproducible Example**.
- Include your system info (OS, Python version, PyTorch version).

## Suggesting Features

We are currently working on **Phase 3 (LLMs)**. If you have specific models or features you'd like to see, please open an issue with the label `enhancement`.

---

**Made with ❤️ for the AI community**
