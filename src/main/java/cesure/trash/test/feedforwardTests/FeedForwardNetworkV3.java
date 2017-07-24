package cesure.trash.test.feedforwardTests;

import cesure.network.NeuralNetworkError;
import cesure.network.activation.ActivationFunction;
import cesure.utils.Matrix;

import java.util.Random;

import static cesure.utils.MatrixMath.*;
import static cesure.utils.MatrixMath.Matrix_mult;

public class FeedForwardNetworkV3 {

    private static final Matrix inputBias = new Matrix(new double[][] {{1}}); // for comfort

    private int inputDimension;
    private int outputDimension;
    private int nbLayers;
    private int hiddenDimension;

    private Matrix[] weights;
    private Matrix[] lastWeightsChange; // init at 0, used for momentum

    private ActivationFunction activation;

    public FeedForwardNetworkV3(int inputDimension, int outputDimension, int nbHiddenLayers, int hiddenDimension, ActivationFunction activation) {
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

    private void weightInit_UNIFORM() {
        for (Matrix weight : weights) {
            weight.setRandom(-1, 1);
        }
    }


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


    public double train_backpropagation(Matrix input, Matrix ideal, double learningRate, double momentum) {
        if (!input.isRowMatrix() || input.nbColumns != inputDimension) {
            throw new NeuralNetworkError("FeedforwardLayer.getOutput(Matrix) - "+input.nbRows+" - "+input.nbColumns +" - "+inputDimension+" - "+outputDimension);
        }
        if (!ideal.isRowMatrix() || ideal.nbColumns != outputDimension) {
            throw new NeuralNetworkError("FeedforwardLayer.getOutput(Matrix) - "+ideal.nbRows+" - "+ideal.nbColumns +" - "+inputDimension+" - "+outputDimension);
        }

        // Forward propagation
        Matrix[] hiddens_unact = new Matrix[nbLayers]; // Unactiavted
        Matrix[] hiddens_act_bias = new Matrix[nbLayers]; // Activated, with bias

        hiddens_unact[0] = input.cp();
        hiddens_act_bias[0] = Matrix_concatenateRowMatrix(hiddens_unact[0], inputBias);
        for (int layerI=1; layerI<nbLayers; layerI++) {
            hiddens_unact[layerI] = Matrix_mDot(hiddens_act_bias[layerI-1], weights[layerI-1]);
            hiddens_act_bias[layerI] = activation.activate(hiddens_unact[layerI]);
            hiddens_act_bias[layerI] = Matrix_concatenateRowMatrix(hiddens_act_bias[layerI], inputBias);
        }
        Matrix output_unact = Matrix_mDot(hiddens_act_bias[nbLayers-1], weights[nbLayers-1]);
        Matrix output_act = activation.activate(output_unact);

        // Backpropagation
        Matrix[] signalsError = new Matrix[nbLayers];
        Matrix[] weightsDelta = new Matrix[nbLayers];

        signalsError[nbLayers-1] = Matrix_substract(output_act, ideal) // signalError[i] -> from layer i to layer i-1
                .pMult(activation.derivative(output_unact));
        weightsDelta[nbLayers-1] = Matrix_mDot(Matrix_transpose(hiddens_act_bias[nbLayers-1]), signalsError[nbLayers-1])
                .mult(-1 * learningRate);
        for (int lay=nbLayers-2; lay>=0; lay--) {
            signalsError[lay] = Matrix_mDot(signalsError[lay+1], Matrix_transpose(weights[lay+1]))
                    .deleteLastColumn() // delete bias because we don't propagate its error
                    .pMult(activation.derivative(hiddens_unact[lay+1]));
            weightsDelta[lay] = Matrix_mDot( Matrix_transpose(hiddens_act_bias[lay]), signalsError[lay] )
                    .mult(-1 * learningRate);
        }

        // Weights update
        for (int lay=0; lay<nbLayers; lay++) {
            weights[lay].add(weightsDelta[lay]).add(Matrix_mult(lastWeightsChange[lay], momentum));
            lastWeightsChange[lay] = weightsDelta[lay];
        }

        return Math.abs(Matrix_substract(ideal, output_act).avg());
    }



    public double train_backpropagation(Matrix[] inputs, Matrix[] ideals, double learningRate, double momentum) {
        if (inputs.length != ideals.length) {
            throw new NeuralNetworkError("FeedforwardNetwork.learn(Matrix[],Matrix[],double,double)");
        }
        for (int i=0; i<inputs.length; i++) {
            if (inputs[i].nbColumns != inputDimension || ideals[i].nbColumns != outputDimension || !inputs[i].isRowMatrix() || !ideals[i].isRowMatrix()) {
                throw new NeuralNetworkError("FeedforwardNetwork.learn(Matrix[],Matrix[],double,double)");
            }
        }

        // Initialization
        int nbPatterns = inputs.length;

        Matrix[] weights_delta_sum = new Matrix[nbLayers];
        for (int layerI=0; layerI<nbLayers; layerI++) {
            weights_delta_sum[layerI] = new Matrix(weights[layerI].nbRows, weights[layerI].nbColumns);
            weights_delta_sum[layerI].setZero();
        }
        double error_sum = 0;

        for (int patternI=0; patternI < nbPatterns; patternI++) {
            // Forward propagation
            Matrix[] hiddens_unact = new Matrix[nbLayers]; // Unactiavted
            Matrix[] hiddens_act_bias = new Matrix[nbLayers]; // Activated, with bias

            hiddens_unact[0] = inputs[patternI].cp();
            hiddens_act_bias[0] = Matrix_concatenateRowMatrix(hiddens_unact[0], inputBias);
            for (int layerI=1; layerI<nbLayers; layerI++) {
                hiddens_unact[layerI] = Matrix_mDot(hiddens_act_bias[layerI-1], weights[layerI-1]);
                hiddens_act_bias[layerI] = activation.activate(hiddens_unact[layerI]);
                hiddens_act_bias[layerI] = Matrix_concatenateRowMatrix(hiddens_act_bias[layerI], inputBias);
            }
            Matrix output_unact = Matrix_mDot(hiddens_act_bias[nbLayers-1], weights[nbLayers-1]);
            Matrix output_act = activation.activate(output_unact);

            // Backpropagation
            Matrix[] signalsError = new Matrix[nbLayers];
            Matrix[] weightsDelta = new Matrix[nbLayers];

            signalsError[nbLayers-1] = Matrix_substract(output_act, ideals[patternI]) // signalError[i] -> from layer i to layer i-1
                    .pMult(activation.derivative(output_unact));
            weightsDelta[nbLayers-1] = Matrix_mDot(Matrix_transpose(hiddens_act_bias[nbLayers-1]), signalsError[nbLayers-1])
                    .mult(-1 * learningRate);
            weights_delta_sum[nbLayers-1].add(weightsDelta[nbLayers-1]);
            for (int layerI=nbLayers-2; layerI>=0; layerI--) {
                signalsError[layerI] = Matrix_mDot(signalsError[layerI+1], Matrix_transpose(weights[layerI+1]))
                        .deleteLastColumn() // delete bias because we don't propagate its error
                        .pMult(activation.derivative(hiddens_unact[layerI+1]));
                weightsDelta[layerI] = Matrix_mDot( Matrix_transpose(hiddens_act_bias[layerI]), signalsError[layerI] )
                        .mult(-1 * learningRate);
                weights_delta_sum[layerI].add(weightsDelta[layerI]);
            }

            error_sum += Math.abs(Matrix_substract(ideals[patternI], output_act).avg());
        }

        // Weights update
        for (int layerI=0; layerI<nbLayers; layerI++) {
            Matrix weightDelta = weights_delta_sum[layerI];
            weights[layerI].add(weightDelta).add(Matrix_mult(lastWeightsChange[layerI], momentum));
            lastWeightsChange[layerI] = weightDelta;
        }

        return error_sum / nbPatterns;
    }

}
