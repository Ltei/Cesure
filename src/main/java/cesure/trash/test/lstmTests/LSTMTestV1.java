package cesure.trash.test.lstmTests;

import cesure.trash.CesureGate;
import cesure.network.activation.ActivationSigmoid;
import cesure.network.activation.ActivationTanh;
import cesure.utils.Matrix;
import cesure.utils.SerializationManager;

import java.io.IOException;
import java.io.Serializable;

import static cesure.utils.MatrixMath.Matrix_concatenateRowMatrix;
import static cesure.utils.MatrixMath.Matrix_pMult;
import static cesure.utils.MatrixMath.Matrix_substract;

public class LSTMTestV1 implements Serializable {
    private final int outputDimension;

    private final int contextDimension;

    private final int noteInputDimension;
    private final int contextInputDimension;

    private CesureGate outputGate;
    private CesureGate forgetGate;
    private CesureGate inputGate;
    private CesureGate rememberGate;

    private Matrix context;


    /****************************************************************
     * Default constructor
     ****************************************************************/
    public LSTMTestV1() {
        outputDimension = 1;

        contextDimension = 8;

        noteInputDimension = contextDimension;
        contextInputDimension = noteInputDimension + outputDimension;

        context = Matrix.newRowMatrix(contextDimension);

        resetContext();

        outputGate = new CesureGate(noteInputDimension, outputDimension, 4,8, new ActivationSigmoid());
        forgetGate = new CesureGate(contextInputDimension, contextDimension, 3,4, new ActivationSigmoid());
        inputGate = new CesureGate(contextInputDimension, contextDimension,3,4, new ActivationSigmoid());
        rememberGate = new CesureGate(contextInputDimension, contextDimension, 3,4, new ActivationTanh());

        Matrix[] ideals = new Matrix[50];
        ideals[0] = Matrix.newRowMatrix(new double[] {0});
        ideals[1] = Matrix.newRowMatrix(new double[] {1});
        ideals[2] = Matrix.newRowMatrix(new double[] {1});
        for (int i=3; i<50; i++) {
            ideals[i] = Matrix.newRowMatrix(new double[] {XOR(ideals[i-3].array[0][0], ideals[i-1].array[0][0])});
        }

        //train_simulatedAnnealing(ideals,4,0.5,100000);
        smartTraining(ideals,4, 100000, 0.001);

        resetContext();

        Matrix[] outputs = new Matrix[75];
        for (int i=0; i<75; i++) {
            if (i < 4) {
                outputs[i] = ideals[i].cp();
                inputNextNote(ideals[i]);
            } else {
                outputs[i] = computeOutput();
            }

            System.out.println("::: IO "+i+" ::: ");
            System.out.println("Output = "+outputs[i].array[0][0]);
            if (i < 50) {
                System.out.println("Ideal = "+ideals[i].array[0][0]);
            }
        }

    }

    private double XOR(double a, double b) {
        return (a > 0.5 && b < 0.5) || (a < 0.5 && b > 0.5) ? 1 : 0;
    }


    /****************************************************************
     * Copy constructor with random changes
     * @param toClone The CesureLSTM object to clone
     * @param randomMagnitude The random changes magnitude
     ****************************************************************/
    public LSTMTestV1(LSTMTestV1 toClone, double randomMagnitude) {
        outputDimension = toClone.outputDimension;

        contextDimension = toClone.contextDimension;

        noteInputDimension = toClone.noteInputDimension;
        contextInputDimension = toClone.contextInputDimension;

        context = toClone.context.cp();

        outputGate = new CesureGate(toClone.outputGate, randomMagnitude);
        forgetGate = new CesureGate(toClone.forgetGate, randomMagnitude);
        inputGate = new CesureGate(toClone.inputGate, randomMagnitude);
        rememberGate = new CesureGate(toClone.rememberGate, randomMagnitude);
    }


    /****************************************************************
     * Compute the next music's note
     * Will change the context according to the outputed note
     * @return The computed note
     ****************************************************************/
    public Matrix computeOutput() {
        // Comppute output
        final Matrix noteInput = context.cp();
        final Matrix outputVect = outputGate.compute(noteInput);

        // Compte context update
        final Matrix contextInput = Matrix_concatenateRowMatrix(noteInput, outputVect);
        final  Matrix forgetVect = forgetGate.compute(contextInput);
        context.pMult(forgetVect);
        final Matrix rememberVect = Matrix_pMult(inputGate.compute(contextInput), rememberGate.compute(contextInput));
        context.add(rememberVect);

        return outputVect;
    }


    /****************************************************************
     * Input the next music's note :
     * Will change the context according to the input note
     * as if it was outputed by a computeNextNote() call
     * @param note The note to input
     ****************************************************************/
    public void inputNextNote(Matrix note) {
        // Compte context update
        final Matrix contextInput = Matrix_concatenateRowMatrix(context.cp(), note);
        final Matrix forgetVect = forgetGate.compute(contextInput);
        context.pMult(forgetVect);
        final Matrix rememberVect = Matrix_pMult(inputGate.compute(contextInput), rememberGate.compute(contextInput));
        context.add(rememberVect);
    }


    /****************************************************************
     * Reset the network's context
     ****************************************************************/
    public void resetContext() {
        context.setZero();
    }


    /****************************************************************
     * Perform a smart simulated annealing training on a set of datas
     * @param ideals The ideal outputs
     * @param start The first note to start guess at :
     *              The "start" first notes will change the context
     *              without being trained on
     * @param maxIterations The maximum iterations
     * @param errorToReach The acceptable error
     ****************************************************************/
    public void smartTraining(Matrix[] ideals, int start, int maxIterations, double errorToReach) {
        double thisError_sum = calculateErrorSum(ideals,start);
        double magnitude = 0.5;
        double magnitudeScaler = 100000;
        double maxMagnitudeScaler = 300000;

        for (int epochI=0; epochI<maxIterations || thisError_sum < errorToReach*ideals.length; epochI++) {

            thisError_sum = calculateErrorSum(ideals,start);

            LSTMTestV1 newNetwork = new LSTMTestV1(this, magnitude*magnitudeScaler/100000);
            double newError_sum = newNetwork.calculateErrorSum(ideals,start);

            if (newError_sum < thisError_sum) {
                outputGate = newNetwork.outputGate;
                forgetGate = newNetwork.forgetGate;
                inputGate = newNetwork.inputGate;
                rememberGate = newNetwork.rememberGate;
                context = newNetwork.context;
                magnitudeScaler += 1;
                if (magnitudeScaler > maxMagnitudeScaler) {
                    magnitudeScaler = 1000;
                }
                thisError_sum = newError_sum;
            } else {
                magnitudeScaler -= 1;
                if (magnitudeScaler < 10) {
                    magnitudeScaler = 100000;
                }
            }

            if (epochI % 100 == 0) {
                System.out.println("Epoch #"+epochI+" - Error = "+thisError_sum+" - MagnitudeScl = "+magnitudeScaler);
            }
        }

    }


    /****************************************************************
     * Perform a simulated annealing training on one music
     * @param ideals The ideal outputs
     * @param start The first note to start guess at :
     *              The "start" first notes will change the context
     *              without being trained on
     * @param magnitude The magnitute
     * @param iterations The number of iterations
     ****************************************************************/
    public void train_simulatedAnnealing(Matrix[] ideals, int start, double magnitude, int iterations) {
        double actualMagnitude;
        for (int epochI=0; epochI<iterations; epochI++) {
            actualMagnitude = (iterations-epochI) * magnitude / iterations;
            double error_sum = simulatedAnnealingIteration(ideals,start, actualMagnitude);
            if (epochI % 100 == 0) {
                System.out.println("Epoch #"+epochI+" - Error = "+error_sum);
            }
        }
    }


    /****************************************************************
     * Perform a single simulated annealing iteration
     * @param ideals The ideal outputs
     * @param start The first note to start guess at :
     *              The "start" first notes will change the context
     *              without being trained on
     * @param magnitude How much to change the weights
     * @return The new network error sum
     ****************************************************************/
    public double simulatedAnnealingIteration(Matrix[] ideals, int start, double magnitude) {
        LSTMTestV1 newNetwork = new LSTMTestV1(this, magnitude);

        double thisError_sum = calculateErrorSum(ideals,start);
        double newError_sum = newNetwork.calculateErrorSum(ideals,start);

        if (newError_sum < thisError_sum) {
            outputGate = newNetwork.outputGate;
            forgetGate = newNetwork.forgetGate;
            inputGate = newNetwork.inputGate;
            rememberGate = newNetwork.rememberGate;
            context = newNetwork.context;
            return newError_sum;
        } else {
            return thisError_sum;
        }
    }


    /****************************************************************
     * Calculate the sum of the errors between this output and a
     * CesureMusic object notes
     * @param ideals The ideal outputs
     * @param start The first output to start add errors at :
     *              The "start" first output's errors won't be added
     *              to the sum
     * @return The error sum
     ****************************************************************/
    public double calculateErrorSum(Matrix[] ideals, int start) {
        int nbOutputs = ideals.length;

        resetContext();
        double errorSum = 0;
        for (int outputI=0; outputI<nbOutputs; outputI++) {
            if (outputI<start) {
                inputNextNote(ideals[outputI]);
            } else {
                errorSum += Math.abs(Matrix_substract(ideals[outputI], computeOutput()).avg());
            }
        }
        return errorSum;
    }


    /****************************************************************
     * Create a new LSTMTestV1 object from a serialized file
     * @param name The serialized file name
     * @return The newly loaded LSTMTestV1 object
     ****************************************************************/
    public static LSTMTestV1 loadFromSerialization(String name) {
        try {
            return (LSTMTestV1) SerializationManager.load(name+".lstm");
        } catch (IOException e) {
            e.printStackTrace();
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
        }
        return null;
    }


    /****************************************************************
     * Write a serialized file from a LSTMTestV1 object
     * @param name The serialized file name
     ****************************************************************/
    public void saveSerialization(String name) {
        try {
            SerializationManager.save(name + ".lstm", this);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

}
