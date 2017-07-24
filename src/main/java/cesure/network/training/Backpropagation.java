package cesure.network.training;

import cesure.network.Cesure;
import cesure.network.CesureGate;
import cesure.network.CesureMusic;
import cesure.utils.Matrix;

import java.util.Random;

import static cesure.utils.MatrixMath.*;

public class Backpropagation {

    /****************************************************************
     * Perform a backpropagation training on a music
     * @param network The Cesure object to train
     * @param music The CesureMusic object to train on
     * @param start The first note to start guess at :
     *              The "start" first notes will change the context
     *              without being trained on
     * @param learningRate The learning rate
     * @param momentum The momentum
     * @param iterations The number of iterations
     ****************************************************************/
    public static void train(Cesure network, CesureMusic music, int start, double learningRate, double momentum, int iterations) {

        Matrix lastChanges; // for momentum

        for (int epochI=0; epochI<iterations; epochI++) {


            /*// Forward propagation
            Matrix[] hiddens_unact = new Matrix[nbLayers]; // Unactivated
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
            }*/
        }

    }

}
