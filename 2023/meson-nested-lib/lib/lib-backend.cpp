#include <memory>
#include "lib.h"

struct MyAwesomeBackend : Backend {
  std::string name() const override {
    return "MyAwesomeBackend";
  }
};

struct RegisterMyAwesomeBackend {
  std::unique_ptr<MyAwesomeBackend> backend;
  RegisterMyAwesomeBackend() : backend(new MyAwesomeBackend()) {
    registerBackend(backend.get());
  }
};

const static RegisterMyAwesomeBackend register_backend;
