package cesure.network.activation;

import static cesure.utils.CesureMath.abs;
import static cesure.utils.CesureMath.square;

public class ActivationTanh extends ActivationFunction {

    public ActivationTanh() {}



    @Override
    public double activate(double x) {
        return x/(1+abs(x));
    }

    @Override
    public double derivative(double x) {
        return 1/square(1+abs(x));
    }



    public double old_activate(double x) {
        return Math.tanh(x);
    }

    public double old_derivative(double x) {
        double coshx = Math.cosh(x);
        double denom = (Math.cosh(2*x) + 1);
        return 4 * coshx * coshx / (denom * denom);
    }
}
