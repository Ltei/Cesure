package cesure.trash;

import cesure.network.NeuralNetworkError;
import cesure.network.activation.ActivationFunction;
import cesure.utils.Matrix;

import java.io.Serializable;
import java.util.Random;

import static cesure.utils.MatrixMath.*;

class CesureGate implements Serializable {

    private static final Matrix inputBias = new Matrix(new double[][] {{1}}); // for comfort

    private int inputDimension;
    private int outputDimension;
    private int nbLayers;
    private int hiddenDimension;

    private Matrix[] weights;
    private Matrix[] lastWeightsChange; // init at 0, used for momentum

    private ActivationFunction activation;

    /****************************************************************
     * Default constructor
     * @param inputDimension The input dimension
     * @param outputDimension The output dimension
     * @param nbHiddenLayers The number of hidden layers
     * @param hiddenDimension The hidden neurons dimension
     * @param activation The activation function
     ****************************************************************/
    public CesureGate(int inputDimension, int outputDimension, int nbHiddenLayers, int hiddenDimension, ActivationFunction activation) {
        nbHiddenLayers = (nbHiddenLayers < 0) ? 0 : nbHiddenLayers;

        this.inputDimension = inputDimension;
        this.outputDimension = outputDimension;
        this.nbLayers = nbHiddenLayers + 1;
        this.hiddenDimension = hiddenDimension;

        weights = new Matrix[nbLayers];
        lastWeightsChange = new Matrix[nbLayers];

        this.activation = activation;

        if (nbHiddenLayers <= 0) { // nbLayers = 1
            weights[0] = new Matrix(inputDimension+1, outputDimension); // +1 for bias
            lastWeightsChange[0] = new Matrix(inputDimension+1, outputDimension); // +1 for bias
        } else { // nbLayers >= 2
            weights[0] = new Matrix(inputDimension+1, hiddenDimension); // +1 for bias
            lastWeightsChange[0] = new Matrix(inputDimension+1, hiddenDimension); // +1 for bias
            for (int i=1; i<nbLayers-1; i++) {
                weights[i] = new Matrix(hiddenDimension+1, hiddenDimension); // +1 for bias
                lastWeightsChange[i] = new Matrix(hiddenDimension+1, hiddenDimension); // +1 for bias
            }
            weights[nbLayers-1] = new Matrix(hiddenDimension+1, outputDimension); // +1 for bias
            lastWeightsChange[nbLayers-1] = new Matrix(hiddenDimension+1, outputDimension); // +1 for bias
        }

        weightInit_XAVIER(new Random());
        for (Matrix lastWeightChange : lastWeightsChange) {
            lastWeightChange.setZero();
        }
    }

    /****************************************************************
     * Copy constructor with an amout of randomization
     * @param toClone The CesureFeedForwardGate object to clone
     * @param randomMagnitude The weights random changes magnitude
     ****************************************************************/
    public CesureGate(CesureGate toClone, double randomMagnitude) {
        inputDimension = toClone.inputDimension;
        outputDimension = toClone.outputDimension;
        nbLayers = toClone.nbLayers;
        hiddenDimension = toClone.hiddenDimension;

        weights = new Matrix[nbLayers];
        lastWeightsChange = new Matrix[nbLayers];

        activation = toClone.activation.clone();

        for (int layerI=0; layerI<nbLayers; layerI++) {
            weights[layerI] = toClone.weights[layerI].cp();
            weights[layerI].randomize(-randomMagnitude, randomMagnitude);
            lastWeightsChange[layerI] = toClone.lastWeightsChange[layerI].cp();
        }
    }


    public CesureGate(CesureGate toClone, Random rand, double randomMagnitude) {
        inputDimension = toClone.inputDimension;
        outputDimension = toClone.outputDimension;
        nbLayers = toClone.nbLayers;
        hiddenDimension = toClone.hiddenDimension;

        weights = new Matrix[nbLayers];
        lastWeightsChange = new Matrix[nbLayers];

        activation = toClone.activation.clone();

        for (int layerI=0; layerI<nbLayers; layerI++) {
            weights[layerI] = toClone.weights[layerI].cp();
            weights[layerI].randomize(rand, -randomMagnitude, randomMagnitude);
            lastWeightsChange[layerI] = toClone.lastWeightsChange[layerI].cp();
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
