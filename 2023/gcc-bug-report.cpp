#include <iostream>
#include <variant>
#include <complex>

void print(std::complex<double> cd) {
  std::cout << "complex: " << cd << std::endl;
}

void print(int64_t i) {
  std::cout << "int64_t: " << i << std::endl;
}

int main() {
  int64_t x[] = {size_t(1)};
  std::complex<double> y[] = {size_t(1)};

  print(size_t(1));
  std::variant<std::complex<double>, int64_t> v = size_t(1);
  if (std::holds_alternative<int64_t>(v)) {
    std::cout << "variant is int" << std::endl;
  } else if (std::holds_alternative<std::complex<double>>(v)) {
    std::cout << "variant is complex" << std::endl;
  }
}