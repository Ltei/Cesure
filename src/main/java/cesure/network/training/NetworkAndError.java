package cesure.network.training;

import cesure.network.Cesure;

public class NetworkAndError {
    public Cesure network;
    public double error_sum;
    public NetworkAndError(Cesure network, double error_sum) {
        this.network = network;
        this.error_sum = error_sum;
    }
}
