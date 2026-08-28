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

std::shared_ptr<Value> operator+(const std::shared_ptr<Value>& lhs, const std::shared_ptr<Value>& rhs){
    
    if(!lhs || !rhs){
        return nullptr;
    }

    std::vector<std::shared_ptr<Value>> prev;
    prev.push_back(lhs);
    prev.push_back(rhs);

    std::shared_ptr<Value> out = std::make_shared<Value>(lhs->data + rhs->data, 0.0, prev, "+");

    out->_backward = [lhs, rhs, out](){
        lhs->grad = out->grad * 1;
        rhs->grad = out->grad * 1;
    };

    return out;
    
}

std::shared_ptr<Value> operator+(const std::shared_ptr<Value>& lhs, double rhs){
    
    if(!lhs){
        return nullptr;
    }

    std::shared_ptr<Value> rhs_ptr = std::make_shared<Value>(rhs, 0.0, std::vector<std::shared_ptr<Value>>{}, "");

    std::vector<std::shared_ptr<Value>> prev;
    prev.push_back(lhs);
    prev.push_back(rhs_ptr);

    std::shared_ptr<Value> out = std::make_shared<Value>(lhs->data + rhs, 0.0, prev, "+");

    out->_backward = [lhs, rhs_ptr, out](){
        lhs->grad = out->grad * 1;
        rhs_ptr->grad = out->grad * 1;
    };  

    return out;

}

std::shared_ptr<Value> operator+(double lhs, const std::shared_ptr<Value>& rhs){

    if(!rhs){
        return nullptr;
    }

    return rhs + lhs;
}

std::shared_ptr<Value> operator-(const std::shared_ptr<Value>& lhs, const std::shared_ptr<Value>& rhs){
    
    if(!lhs || !rhs){
        return nullptr;
    }

    std::vector<std::shared_ptr<Value>> prev;
    prev.push_back(lhs);
    prev.push_back(rhs);

    std::shared_ptr<Value> out = std::make_shared<Value>(lhs->data - rhs->data, 0.0, prev, "-");
    out->_backward = [lhs, rhs, out](){
        lhs->grad = out->grad * 1;
        rhs->grad = out->grad * -1;
    };

    return out;
}

std::shared_ptr<Value> operator-(const std::shared_ptr<Value>& lhs, double rhs){
    
    if(!lhs){
        return nullptr;
    }

    std::shared_ptr<Value> rhs_ptr = std::make_shared<Value>(rhs, 0.0, std::vector<std::shared_ptr<Value>>{}, "");

    std::vector<std::shared_ptr<Value>> prev;
    prev.push_back(lhs);
    prev.push_back(rhs_ptr);

    std::shared_ptr<Value> out = std::make_shared<Value>(lhs->data - rhs, 0.0, prev, "-");
    out->_backward = [lhs, rhs_ptr, out](){
        lhs->grad = out->grad * 1;
        rhs_ptr->grad = out->grad * -1;
    };

    return out;
}

std::shared_ptr<Value> operator-(double lhs, const std::shared_ptr<Value>& rhs){

    if(!rhs){
        return nullptr;
    }

    std::shared_ptr<Value> lhs_ptr = std::make_shared<Value>(lhs, 0.0, std::vector<std::shared_ptr<Value>>{}, "");

    return lhs_ptr - rhs;

}

std::shared_ptr<Value> operator*(const std::shared_ptr<Value>& lhs, const std::shared_ptr<Value>& rhs){
    
    if(!lhs || !rhs){
        return nullptr;
    }

    std::vector<std::shared_ptr<Value>> prev;
    prev.push_back(lhs);
    prev.push_back(rhs);

    std::shared_ptr<Value> out = std::make_shared<Value>(lhs->data * rhs->data, 0.0, prev, "*");
    out->_backward = [lhs, rhs, out](){
        lhs->grad = rhs->data * out->grad;
        rhs->grad = lhs->data * out->grad;
    };

    return out;
    
}

std::shared_ptr<Value> operator*(const std::shared_ptr<Value>& lhs, double rhs){
    
    if(!lhs){
        return nullptr;
    }

    std::shared_ptr<Value> rhs_ptr = std::make_shared<Value>(rhs, 0.0, std::vector<std::shared_ptr<Value>>{}, "");

    std::vector<std::shared_ptr<Value>> prev;
    prev.push_back(lhs);
    prev.push_back(rhs_ptr);

    std::shared_ptr<Value> out = std::make_shared<Value>(lhs->data * rhs, 0.0, prev, "*");
    out->_backward = [lhs, rhs_ptr, out](){
        lhs->grad = rhs_ptr->data * out->grad;
        rhs_ptr->grad = lhs->data * out->grad;
    };

    return out;

}

std::shared_ptr<Value> operator*(double lhs, const std::shared_ptr<Value>& rhs){

    if(!rhs){
        return nullptr;
    }

    return rhs * lhs;

}

std::shared_ptr<Value> operator/(const std::shared_ptr<Value>& lhs, const std::shared_ptr<Value>& rhs){
    
    if((!lhs || !rhs)){
        return nullptr;
    }

    std::vector<std::shared_ptr<Value>> prev;
    prev.push_back(lhs);
    prev.push_back(rhs);

    return std::make_shared<Value>(lhs->data / rhs->data, 0.0, prev, "/");
    
}

std::shared_ptr<Value> operator/(const std::shared_ptr<Value>& lhs, double rhs){
    
    if(!lhs){
        return nullptr;
    }

    std::shared_ptr<Value> rhs_ptr = std::make_shared<Value>(rhs, 0.0, std::vector<std::shared_ptr<Value>>{}, "");

    std::vector<std::shared_ptr<Value>> prev;
    prev.push_back(lhs);
    prev.push_back(rhs_ptr);

    return std::make_shared<Value>(lhs->data / rhs, 0.0, prev, "/");

}

std::shared_ptr<Value> operator/(double lhs, const std::shared_ptr<Value>& rhs){

    if(!rhs){
        return nullptr;
    }

    std::shared_ptr<Value> lhs_ptr = std::make_shared<Value>(lhs, 0.0, std::vector<std::shared_ptr<Value>>{}, "");

    return lhs_ptr / rhs;

}



