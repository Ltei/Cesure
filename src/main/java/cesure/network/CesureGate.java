package cesure.network;

import cesure.network.activation.ActivationFunction;
import cesure.utils.Matrix;

import java.io.Serializable;
import java.util.Random;

import static cesure.utils.MatrixMath.Matrix_concatenateRowMatrix;
import static cesure.utils.MatrixMath.Matrix_mDot;

public class CesureGate implements Serializable {

    private static final Matrix inputBias = new Matrix(new double[][] {{1}}); // for comfort

    private final int inputDimension;
    private final int outputDimension;
    private final int nbLayers;

    private final Matrix[] weights;

    private final ActivationFunction activation;

    /****************************************************************
     * Default constructor
     * @param inputDimension The input dimension
     * @param outputDimension The output dimension
     * @param activation The activation function
     * @param hiddenDimensions The hidden layers dimensions, there is
     *                         hiddenDimensions.length hidden layers
     ****************************************************************/
    public CesureGate(final int inputDimension, final int outputDimension, final ActivationFunction activation, final int... hiddenDimensions) {
        if (inputDimension < 1 || outputDimension < 1) {
            throw new NeuralNetworkError("CesureGate2(int,int,ActivationFunction,int)");
        }
        for (int i : hiddenDimensions) {
            if (i < 1) {
                throw new NeuralNetworkError("CesureGate2(int,int,ActivationFunction,int)");
            }
        }

        final int nbHiddenLayers = hiddenDimensions.length;

        this.inputDimension = inputDimension;
        this.outputDimension = outputDimension;
        this.nbLayers = hiddenDimensions.length + 1;

        this.activation = activation;

        weights = new Matrix[nbLayers];

        if (nbLayers == 1) { // nbLayers = 1
            weights[0] = new Matrix(inputDimension+1, outputDimension); // +1 for bias
        } else { // nbLayers >= 2
            weights[0] = new Matrix(inputDimension+1, hiddenDimensions[0]); // +1 for bias
            for (int i=1; i<nbHiddenLayers; i++) {
                weights[i] = new Matrix(hiddenDimensions[i-1]+1, hiddenDimensions[i]); // +1 for bias
            }
            weights[nbLayers-1] = new Matrix(hiddenDimensions[nbHiddenLayers-1]+1, outputDimension); // +1 for bias
        }

        weightInit_XAVIER(new Random());
    }

    public static CesureGate newAutoCesureGate2(final int inputDimension, final int outputDimension, final ActivationFunction activation, final int nbHiddenLayers) {
        if (nbHiddenLayers < 0) {
            throw new NeuralNetworkError("CesureGate2.newAutoCesureGate2(int,int,ActivationFunction,int)");
        }

        final int[] hiddenDimensions = new int[nbHiddenLayers];
        for (int i=0; i<nbHiddenLayers; i++) {
            double x = (i+1.0) / (nbHiddenLayers+1.0);
            hiddenDimensions[i] = (int) Math.round(x*outputDimension + (1-x)*inputDimension);
        }

        return new CesureGate(inputDimension,outputDimension,activation,hiddenDimensions);
    }

    /****************************************************************
     * Copy constructor with an amount of randomization
     * @param toClone The CesureFeedForwardGate object to clone
     * @param randomMagnitude The weights random changes magnitude
     ****************************************************************/
    public CesureGate(CesureGate toClone, double randomMagnitude) {
        inputDimension = toClone.inputDimension;
        outputDimension = toClone.outputDimension;
        nbLayers = toClone.nbLayers;

        weights = new Matrix[nbLayers];

        activation = toClone.activation.clone();

        for (int layerI=0; layerI<nbLayers; layerI++) {
            weights[layerI] = toClone.weights[layerI].cp();
            weights[layerI].randomize(-randomMagnitude, randomMagnitude);
        }
    }


    /****************************************************************
     * Copy constructor with an amount of randomization, using the
     * random of a Random object
     * @param cloned The CesureGate object to clone
     * @param rand The Random object
     * @param magnitude The weights random changes magnitude
     ****************************************************************/
    public CesureGate(CesureGate cloned, Random rand, double magnitude) {
        inputDimension = cloned.inputDimension;
        outputDimension = cloned.outputDimension;
        nbLayers = cloned.nbLayers;

        weights = new Matrix[nbLayers];

        activation = cloned.activation.clone();

        for (int layerI=0; layerI<nbLayers; layerI++) {
            weights[layerI] = cloned.weights[layerI].cp();
            weights[layerI].randomize(rand, -magnitude, magnitude);
        }
    }




    /****************************************************************
     * Initialize weights using XAVIER's initialization
     ****************************************************************/
    private void weightInit_XAVIER(Random rand) {
        for (Matrix weight : weights) {
            double stdDeviation = 2.0 / weight.nbRows;
            for (int row=0; row<weight.nbRows; row++) {
                for (int col=0; col<weight.nbColumns; col++) {
                    weight.array[row][col] = rand.nextGaussian() * stdDeviation;
                }
            }
        }
    }

    /****************************************************************
     * Initialize weights using uniform random on [-1,1]
     ****************************************************************/
    private void weightInit_UNIFORM() {
        for (Matrix weight : weights) {
            weight.setRandom(-1, 1);
        }
    }


    /****************************************************************
     * Compute an input
     * @param input : The input to compute
     * @return The computed output
     ****************************************************************/
    public Matrix compute(Matrix input) {
        if (!input.isRowMatrix() || input.nbColumns != inputDimension) {
            throw new NeuralNetworkError("FeedforwardLayer.getOutput(Matrix) - "+input.nbRows+" - "+input.nbColumns
                    +" - "+inputDimension+" - "+outputDimension);
        }

        Matrix hidden = Matrix_concatenateRowMatrix(input, inputBias);
        for (int i=0; i<nbLayers-1; i++) {
            hidden = activation.activate( Matrix_mDot(hidden, weights[i]) );
            hidden = Matrix_concatenateRowMatrix(hidden, inputBias);
        }
        Matrix output = activation.activate( Matrix_mDot(hidden, weights[nbLayers-1]) );
        return output;
    }

    public CesureGateComputeInfos computeAndGetInfos(Matrix input) {
        if (!input.isRowMatrix() || input.nbColumns != inputDimension) {
            throw new NeuralNetworkError("FeedforwardLayer.getOutput(Matrix) - "+input.nbRows+" - "+input.nbColumns
                    +" - "+inputDimension+" - "+outputDimension);
        }

        final Matrix[] hiddens_unact = new Matrix[nbLayers];
        final Matrix[] hiddens = new Matrix[nbLayers];

        hiddens_unact[0] = input;
        hiddens[0] = input;
        for (int i=0; i<nbLayers-1; i++) {
            final Matrix in = Matrix_concatenateRowMatrix(hiddens[i], inputBias);
            hiddens_unact[i+1] = Matrix_mDot(in, weights[i]);
            hiddens[i+1] = activation.activate(hiddens_unact[i+1]);
        }
        final Matrix in = Matrix_concatenateRowMatrix(hiddens[nbLayers-1], inputBias);
        final Matrix output_unact = Matrix_mDot(in, weights[nbLayers-1]);
        final Matrix output = activation.activate(output_unact);

        return new CesureGateComputeInfos(hiddens_unact, hiddens, output_unact, output);
    }



    /****************************************************************
     * Return the total number of neurons in this gate
     * @return The total number of neurons in this gate
     ****************************************************************/
    public int getNbNeurons() {
        int count = 0;
        for (Matrix weight : weights) {
            count += weight.nbRows * weight.nbColumns;
        }
        return count;
    }


    /****************************************************************
     * Print the layers
     ****************************************************************/
    public void print() {
        for (int i=0; i<nbLayers; i++) {
            System.out.println("Layer #"+i+" :");
            weights[i].print();
        }
    }

}
