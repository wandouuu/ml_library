#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

class Value : public std::enable_shared_from_this<Value> {
    private:
        double data = 0.0; // holding the value of tensor
        double grad = 0.0; // gradient

        std::vector<std::shared_ptr<Value>> prev; // vector of pointers towards parent nodes

        std::string op; // operand
        std::function<void()> _backward{[](){}}; // backward function
    
    public:
        Value(double data = 0.0, 
              double grad = 0.0,
              std::vector<std::shared_ptr<Value>> prev = {}, 
              std::string op = "");

        ~Value() = default;

        double get_data() const; // keep content of object constant
        double get_grad() const; // same as above
        const std::vector<std::shared_ptr<Value>>& get_prev() const; // keep content of vector constant but also prevent memory to be changed
        const std::string& get_op() const; // similar to above
        const std::function<void()>& get_backward() const; // similar to above


};
