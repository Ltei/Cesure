package cesure.network.activation;

public class ActivationIdentity extends ActivationFunction {

    public ActivationIdentity() {}

    @Override
    public double activate(double x) {
        return x;
    }

    @Override
    public double derivative(double x) {
        return 1;
    }
}
