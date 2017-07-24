package cesure.network.activation;

import static cesure.utils.CesureMath.abs;
import static cesure.utils.CesureMath.square;

public class ActivationSigmoid extends ActivationFunction {

    public ActivationSigmoid() {}



    @Override
    public double activate(double x) {
        return 0.5 - 0.5*(x/(1+abs(x)));
    }

    @Override
    public double derivative(double x) {
        return -0.5/square(1+abs(x));
    }



    public double old_activate(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    public double old_derivative(double x) {
        double act = activate(x);
        return act * (1 - act);
    }
}
