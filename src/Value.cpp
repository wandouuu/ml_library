#include "value.hpp"

Value::Value(double data, double grad, std::vector<std::shared_ptr<Value>> prev, std::string op) : 
    data(data), 
    grad(0.0), 
    prev(std::move(prev)), 
    op(std::move(op)){}

double Value::get_data() const{
    return data;
}

double Value::get_grad() const{
    return grad;
}

const std::vector<std::shared_ptr<Value>>& Value::get_prev() const{
    return prev;
}

const std::string& Value::get_op() const{ 
    return op;
}

const std::function<void()>& Value::get_backward() const{
    return _backward;
}