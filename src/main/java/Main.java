import cesure.network.FeedForwardNetworkV4;
import cesure.network.activation.ActivationSigmoid;
import cesure.utils.Matrix;

public class Main {

    public static void main(String[] args) {

        FeedForwardNetworkV4 network = new FeedForwardNetworkV4(2, 1, 1, 3, new ActivationSigmoid());

        Matrix[] inputs = new Matrix[] {
                Matrix.newRowMatrix(new double[] {0,0}),
                Matrix.newRowMatrix(new double[] {0,1}),
                Matrix.newRowMatrix(new double[] {1,0}),
                Matrix.newRowMatrix(new double[] {1,1})
        };

        Matrix[] ideals = new Matrix[] {
                Matrix.newRowMatrix(new double[] {0}),
                Matrix.newRowMatrix(new double[] {1}),
                Matrix.newRowMatrix(new double[] {1}),
                Matrix.newRowMatrix(new double[] {0})
        };

        for (int i=0; i<1000000; i++) {
            if (i%1000 == 0) {
                System.out.println( i+" - "+network.train_backpropagation(inputs, ideals, .5, .9) );
            }

        }

        network.compute(inputs[0]).print();
        network.compute(inputs[1]).print();
        network.compute(inputs[2]).print();
        network.compute(inputs[3]).print();
    }

}
