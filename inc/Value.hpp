#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>
#include <set>

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
        
        // Getters
        double get_data() const; // keep content of object constant
        double get_grad() const; // same as above
        const std::vector<std::shared_ptr<Value>>& get_prev() const; // keep content of vector constant but also prevent memory to be changed
        const std::string& get_op() const; // similar to above
        const std::function<void()>& get_backward() const; // similar to above
        
        // For testing ONLY
        void set_grad(double grad);

        // Backward pass function
        void backward();

        friend std::shared_ptr<Value> operator+(const std::shared_ptr<Value>& lhs,
                                                const std::shared_ptr<Value>& rhs);
        friend std::shared_ptr<Value> operator+(const std::shared_ptr<Value>& lhs,
                                                double rhs);
        friend std::shared_ptr<Value> operator+(double lhs,
                                                const std::shared_ptr<Value>& rhs);
        
        friend std::shared_ptr<Value> operator-(const std::shared_ptr<Value>& lhs,
                                                const std::shared_ptr<Value>& rhs);
        friend std::shared_ptr<Value> operator-(const std::shared_ptr<Value>& lhs,
                                                double rhs);
        friend std::shared_ptr<Value> operator-(double lhs,
                                                const std::shared_ptr<Value>& rhs);
                                                
        friend std::shared_ptr<Value> operator*(const std::shared_ptr<Value>& lhs,
                                                const std::shared_ptr<Value>& rhs);
        friend std::shared_ptr<Value> operator*(const std::shared_ptr<Value>& lhs,
                                                double rhs);
        friend std::shared_ptr<Value> operator*(double lhs,
                                                const std::shared_ptr<Value>& rhs);
        
        friend std::shared_ptr<Value> operator/(const std::shared_ptr<Value>& lhs,
                                                const std::shared_ptr<Value>& rhs);
        friend std::shared_ptr<Value> operator/(const std::shared_ptr<Value>& lhs,
                                                double rhs);
        friend std::shared_ptr<Value> operator/(double lhs,
                                                const std::shared_ptr<Value>& rhs); 


};

void build_topo(std::vector<std::shared_ptr<Value>>& topo, std::set<std::shared_ptr<Value>>& visited, std::shared_ptr<Value> v);
