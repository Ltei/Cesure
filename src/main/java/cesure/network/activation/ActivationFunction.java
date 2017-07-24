package cesure.network.activation;

import cesure.network.NeuralNetworkError;
import cesure.utils.Matrix;

import java.io.Serializable;

public abstract class ActivationFunction implements Serializable {

    public ActivationFunction clone() {
        if (this instanceof ActivationSigmoid) {
            return new ActivationSigmoid();
        } else if (this instanceof ActivationTanh) {
            return new ActivationTanh();
        } else if (this instanceof ActivationIdentity) {
            return new ActivationIdentity();
        } else {
            throw new NeuralNetworkError("ActivationFunction.clone()");
        }
    }

    public Matrix activate(Matrix x) {
        double[][] newArray = new double[x.nbRows][x.nbColumns];
        for (int i=0; i<x.nbRows; i++) {
            for (int j=0; j<x.nbColumns; j++) {
                newArray[i][j] = activate(x.array[i][j]);
            }
        }
        return new Matrix(newArray);
    }

    public Matrix derivative(Matrix x) {
        double[][] newArray = new double[x.nbRows][x.nbColumns];
        for (int i=0; i<x.nbRows; i++) {
            for (int j=0; j<x.nbColumns; j++) {
                newArray[i][j] = derivative(x.array[i][j]);
            }
        }
        return new Matrix(newArray);
    }

    public abstract double activate(double x);
    public abstract double derivative(double x);

}
