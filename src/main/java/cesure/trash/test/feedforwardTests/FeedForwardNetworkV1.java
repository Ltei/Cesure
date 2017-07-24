package cesure.trash.test.feedforwardTests;

import cesure.network.NeuralNetworkError;
import cesure.utils.Matrix;

import java.util.Random;

import static cesure.utils.MatrixMath.*;

public class FeedForwardNetworkV1 {

    private int inputDimension;
    private int outputDimension;
    private int nbLayers;
    private int hiddenDimension;

    private Matrix[] weights;
    private Matrix[] lastWeightsChange; // init at 0, used for momentum

    //private ActivationFunction activation;

    private Matrix inputBias; // for comfort

    public FeedForwardNetworkV1(int inputDimension, int outputDimension, int nbHiddenLayers, int hiddenDimension) {
        nbHiddenLayers = (nbHiddenLayers < 0) ? 0 : nbHiddenLayers;

        this.inputDimension = inputDimension;
        this.outputDimension = outputDimension;
        this.nbLayers = nbHiddenLayers + 1;
        this.hiddenDimension = hiddenDimension;

        weights = new Matrix[nbLayers];
        lastWeightsChange = new Matrix[nbLayers];

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

        //activation = new ActivationSigmoid();

        inputBias = new Matrix(new double[][] {{1}});
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


    public Matrix compute(Matrix input) {
        if (!input.isRowMatrix() || input.nbColumns != inputDimension) {
            throw new NeuralNetworkError("FeedforwardLayer.getOutput(Matrix) - "+input.nbRows+" - "+input.nbColumns
                    +" - "+inputDimension+" - "+outputDimension);
        }

        Matrix hidden = Matrix_concatenateRowMatrix(input, inputBias);
        for (int i=0; i<nbLayers-1; i++) {
            hidden = sigmoid( Matrix_mDot(hidden, weights[i]) );
            hidden = Matrix_concatenateRowMatrix(hidden, inputBias);
        }
        Matrix output = sigmoid( Matrix_mDot(hidden, weights[nbLayers-1]) );
        return output;
    }



    public double learn(Matrix input, Matrix ideal, double learningRate, double momentum) {
        if (!input.isRowMatrix() || input.nbColumns != inputDimension) {
            throw new NeuralNetworkError("FeedforwardLayer.getOutput(Matrix) - "+input.nbRows+" - "+input.nbColumns +" - "+inputDimension+" - "+outputDimension);
        }
        if (!ideal.isRowMatrix() || ideal.nbColumns != outputDimension) {
            throw new NeuralNetworkError("FeedforwardLayer.getOutput(Matrix) - "+ideal.nbRows+" - "+ideal.nbColumns +" - "+inputDimension+" - "+outputDimension);
        }

        // Forward propagation
        Matrix[] hiddens = new Matrix[nbLayers]; // hiddens[0] = input with bias
        hiddens[0] = Matrix_concatenateRowMatrix(input, inputBias);
        for (int i=1; i<nbLayers; i++) {
            hiddens[i] = sigmoid( Matrix_mDot(hiddens[i-1], weights[i-1]) );
            hiddens[i] = Matrix_concatenateRowMatrix(hiddens[i], inputBias);
        }
        Matrix finalOutput = sigmoid( Matrix_mDot(hiddens[nbLayers-1], weights[nbLayers-1]) );

        // Backpropagation
        Matrix[] signalsError = new Matrix[nbLayers];
        Matrix[] weightsDelta = new Matrix[nbLayers];

        signalsError[nbLayers-1] = Matrix_substract(finalOutput, ideal) // signalError[i] -> from layer i to layer i-1
                .pMult(sigmoidDeriv(finalOutput));
        weightsDelta[nbLayers-1] = Matrix_mDot(Matrix_transpose(hiddens[nbLayers-1]), signalsError[nbLayers-1])
                .mult(-1 * learningRate);

        for (int lay=nbLayers-2; lay>=0; lay--) {
            signalsError[lay] = Matrix_mDot(signalsError[lay+1], Matrix_transpose(weights[lay+1]))
                    .pMult(sigmoidDeriv(hiddens[lay+1]))
                    .deleteLastColumn(); // delete bias because we don't propagate its error
            weightsDelta[lay] = Matrix_mDot( Matrix_transpose(hiddens[lay]), signalsError[lay] )
                    .mult(-1 * learningRate);
        }

        for (int lay=0; lay<nbLayers; lay++) {
            weights[lay].add(weightsDelta[lay]).add(Matrix_mult(lastWeightsChange[lay], momentum));
            lastWeightsChange[lay] = weightsDelta[lay];
        }

        return Math.abs(Matrix_substract(ideal, finalOutput).avg());
    }


    public double learn(Matrix[] inputs, Matrix[] ideals, double learningRate, double momentum) {
        if (inputs.length != ideals.length) {
            throw new NeuralNetworkError("FeedforwardNetwork.learn(Matrix[],Matrix[],double,double)");
        }
        for (int i=0; i<inputs.length; i++) {
            if (inputs[i].nbColumns != inputDimension || ideals[i].nbColumns != outputDimension || !inputs[i].isRowMatrix() || !ideals[i].isRowMatrix()) {
                throw new NeuralNetworkError("FeedforwardNetwork.learn(Matrix[],Matrix[],double,double)");
            }
        }

        int nbPatterns = inputs.length;

        Matrix[] weights_delta_sum = new Matrix[nbLayers];
        for (int layerI=0; layerI<nbLayers; layerI++) {
            weights_delta_sum[layerI] = new Matrix(weights[layerI].nbRows, weights[layerI].nbColumns);
            weights_delta_sum[layerI].setZero();
        }
        double error_sum = 0;

        for (int patternI=0; patternI < nbPatterns; patternI++) {
            Matrix[] hiddens = new Matrix[nbLayers]; // hiddens[0] = input with bias
            hiddens[0] = Matrix_concatenateRowMatrix(inputs[patternI], inputBias);
            for (int i=1; i<nbLayers; i++) {
                hiddens[i] = sigmoid( Matrix_mDot(hiddens[i-1], weights[i-1]) );
                hiddens[i] = Matrix_concatenateRowMatrix(hiddens[i], inputBias);
            }
            Matrix finalOutput = sigmoid( Matrix_mDot(hiddens[nbLayers-1], weights[nbLayers-1]) );

            Matrix[] signalsError = new Matrix[nbLayers];
            Matrix[] weightsDelta = new Matrix[nbLayers];

            signalsError[nbLayers-1] = Matrix_substract(finalOutput, ideals[patternI]) // signalError[i] -> from layer i to layer i-1
                    .pMult(sigmoidDeriv(finalOutput));
            weightsDelta[nbLayers-1] = Matrix_mDot(Matrix_transpose(hiddens[nbLayers-1]), signalsError[nbLayers-1])
                    .mult(-1 * learningRate);
            weights_delta_sum[nbLayers-1].add(weightsDelta[nbLayers-1]);
            for (int layerI=nbLayers-2; layerI>=0; layerI--) {
                signalsError[layerI] = Matrix_mDot(signalsError[layerI+1], Matrix_transpose(weights[layerI+1]))
                        .pMult(sigmoidDeriv(hiddens[layerI+1]))
                        .deleteLastColumn(); // delete bias because we don't propagate its error
                weightsDelta[layerI] = Matrix_mDot( Matrix_transpose(hiddens[layerI]), signalsError[layerI] )
                        .mult(-1 * learningRate);
                weights_delta_sum[layerI].add(weightsDelta[layerI]);
            }

            error_sum += Math.abs(Matrix_substract(ideals[patternI], finalOutput).avg());
        }

        for (int layerI=0; layerI<nbLayers; layerI++) {
            Matrix weightDelta = weights_delta_sum[layerI];
            weights[layerI].add(weightDelta).add(Matrix_mult(lastWeightsChange[layerI], momentum));
            lastWeightsChange[layerI] = weightDelta;
        }

        return error_sum / nbPatterns;
    }



    private double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    private double sigmoidDeriv(double sigmoidX) {
        return sigmoidX * (1 - sigmoidX);
    }

    private Matrix sigmoid(Matrix sigmoidX) {
        Matrix newMatrix = new Matrix(sigmoidX.array);
        for (int row=0; row<newMatrix.nbRows; row++) {
            for (int col=0; col<newMatrix.nbColumns; col++) {
                newMatrix.array[row][col] = sigmoid(newMatrix.array[row][col]);
            }
        }
        return newMatrix;
    }

    private Matrix sigmoidDeriv(Matrix sigmoidX) {
        Matrix newMatrix = new Matrix(sigmoidX.array);
        for (int row=0; row<newMatrix.nbRows; row++) {
            for (int col=0; col<newMatrix.nbColumns; col++) {
                newMatrix.array[row][col] = sigmoidDeriv(newMatrix.array[row][col]);
            }
        }
        return newMatrix;
    }

}




/*public double learn(Matrix input, Matrix ideal, double learningRate, double momentum) {
        if (!input.isRowMatrix() || input.nbColumns != inputDimension) {
            throw new NeuralNetworkError("FeedforwardLayer.getOutput(Matrix) - "+input.nbRows+" - "+input.nbColumns +" - "+inputDimension+" - "+outputDimension);
        }
        if (!ideal.isRowMatrix() || ideal.nbColumns != outputDimension) {
            throw new NeuralNetworkError("FeedforwardLayer.getOutput(Matrix) - "+ideal.nbRows+" - "+ideal.nbColumns +" - "+inputDimension+" - "+outputDimension);
        }


        // Forward propagation
        Matrix[] hiddens = new Matrix[nbLayers]; // hiddens[0] = input with bias
        hiddens[0] = Matrix_concatenateRowMatrix(input, inputBias);
        for (int i=1; i<nbLayers; i++) {
            hiddens[i] = sigmoid( Matrix_mDot(hiddens[i-1], weights[i-1]) );
            hiddens[i] = Matrix_concatenateRowMatrix(hiddens[i], inputBias);
        }
        Matrix finalOutput = sigmoid( Matrix_mDot(hiddens[nbLayers-1], weights[nbLayers-1]) );

        // Backpropagation
        Matrix[] signalsError = new Matrix[nbLayers];
        Matrix[] weightsDelta = new Matrix[nbLayers];
        Matrix[] newWeights = new Matrix[nbLayers];

        signalsError[nbLayers-1] = Matrix_substract(ideal, finalOutput).pMult(sigmoidDeriv(finalOutput)); // signalError[i] -> from layer i to layer i-1
        weightsDelta[nbLayers-1] = Matrix_mDot(Matrix_transpose(hiddens[nbLayers-1]), signalsError[nbLayers-1]) .mult(learningRate);
        newWeights[nbLayers-1] = Matrix_add(weights[nbLayers-1], weightsDelta[nbLayers-1]);
        //newWeights[nbLayers-1].add(Matrix_mult(lastWeightsChange[nbLayers-1], momentum));

        for (int lay=nbLayers-2; lay>=0; lay--) {
            signalsError[lay] = Matrix_mDot(signalsError[lay+1], Matrix_transpose(weights[lay+1]))
                    .pMult(sigmoidDeriv(hiddens[lay+1]))
                    .deleteLastColumn(); // delete bias because we don't propagate its error
            weightsDelta[lay] = Matrix_mDot( Matrix_transpose(hiddens[lay]), signalsError[lay] ).mult(learningRate);
            newWeights[lay] = Matrix_add(weights[lay], weightsDelta[lay]);
            //newWeights[lay].add(Matrix_mult(lastWeightsChange[lay], momentum));
        }

        for (int lay=0; lay<nbLayers; lay++) {
            //weightsDelta[lay].print("Layer"+lay+" delta :");
            lastWeightsChange[lay] = weightsDelta[lay];
            weights[lay] = newWeights[lay];
        }

        return Math.abs(Matrix_substract(ideal, finalOutput).moyenne());
    }*/