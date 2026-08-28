#include "value.hpp"
#include <iostream>

int main(){

    std::shared_ptr<Value> a = std::make_shared<Value>(2);
    std::shared_ptr<Value> b = std::make_shared<Value>(3);

    std::shared_ptr<Value> c = a / b;

    c->set_grad(1);

    std::function<void()> backward_func = c->get_backward();
    backward_func();

    return 0;
}