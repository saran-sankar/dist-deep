# dist-deep

`dist-deep` is a deep learning library implemented in C, designed for distributed training using MPI (Message Passing Interface) and OpenMP (Open Multi-Processing). This library allows users to define, train, and evaluate deep learning models efficiently on distributed systems.

## Features

- **Layer Abstraction**: Support for various layer types, including dense layers.
- **Distributed Training**: Utilizes MPI for parallel processing across multiple nodes.
- **Backpropagation**: Implements backpropagation for updating weights and biases.
- **Flexible Configuration**: Easy to configure models with customizable parameters.

## Installation

To get started with `dist-deep`, clone the repository and compile the source code.

```bash
git clone https://github.com/saran-sankar/dist-deep.git
cd dist-deep
make
```

## Usage

### Example

`example.c`: Demonstrates how to load the Iris dataset, configure a model, and train it using the dist-deep library.

### File Structure

- `include/`
  - `bprop.h`: Contains functions for backpropagation.
  - `estimator.h`: Contains the model training functions.
  - `layers.h`: Defines the structure of layers and activation functions.
  - `fprop.h`: Contains functions for forward propagation.

## Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests for enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [OpenMP](https://www.openmp.org/)
- [MPI](https://www.mpi-forum.org/)
