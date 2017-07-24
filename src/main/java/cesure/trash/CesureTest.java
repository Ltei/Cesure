package cesure.trash;

import cesure.network.*;
import cesure.network.activation.ActivationSigmoid;
import cesure.network.activation.ActivationTanh;
import cesure.utils.Matrix;
import cesure.utils.SerializationManager;

import java.io.IOException;
import java.io.Serializable;
import java.util.Random;

import static cesure.utils.MatrixMath.Matrix_concatenateRowMatrix;
import static cesure.utils.MatrixMath.Matrix_pMult;
import static cesure.utils.MatrixMath.Matrix_substract;

class CesureTest implements Serializable {

    public static final int NB_NOTES;
    public static final int NB_OCTAVES;

    public static final int CHORD_DIMENSION;

    public static final int INFOS_DIMENSION;
    public static final int CONTEXT_DIMENSION;

    public static final int OUTPUTGATE_INPUT_SIZE;
    public static final int OUTPUTGATE_OUTPUT_SIZE;

    public static final int CONTEXTGATES_INPUT_SIZE;
    public static final int CONTEXTGATES_OUTPUT_SIZE;

    static {
        NB_NOTES = 12;
        NB_OCTAVES = 4;

        CHORD_DIMENSION = NB_NOTES * NB_OCTAVES;

        INFOS_DIMENSION = 5;
        CONTEXT_DIMENSION = CHORD_DIMENSION;

        OUTPUTGATE_INPUT_SIZE = INFOS_DIMENSION + CONTEXT_DIMENSION;
        OUTPUTGATE_OUTPUT_SIZE = CHORD_DIMENSION;

        CONTEXTGATES_INPUT_SIZE = INFOS_DIMENSION + CONTEXT_DIMENSION + OUTPUTGATE_OUTPUT_SIZE;
        CONTEXTGATES_OUTPUT_SIZE = CONTEXT_DIMENSION;
    }


    private cesure.network.CesureGate outputGate;
    private cesure.network.CesureGate forgetGate;
    private cesure.network.CesureGate memoryGate;
    private cesure.network.CesureGate memoryInputGate;

    private Matrix context;
    private Matrix infos;


    /****************************************************************
     * Default constructor
     ****************************************************************/
    public CesureTest() {
        outputGate = cesure.network.CesureGate.newAutoCesureGate2(OUTPUTGATE_INPUT_SIZE, OUTPUTGATE_OUTPUT_SIZE, new ActivationSigmoid(), 4);
        forgetGate = cesure.network.CesureGate.newAutoCesureGate2(CONTEXTGATES_INPUT_SIZE, CONTEXTGATES_OUTPUT_SIZE, new ActivationSigmoid(), 4);
        memoryGate = cesure.network.CesureGate.newAutoCesureGate2(CONTEXTGATES_INPUT_SIZE, CONTEXTGATES_OUTPUT_SIZE, new ActivationSigmoid(), 4);
        memoryInputGate = cesure.network.CesureGate.newAutoCesureGate2(CONTEXTGATES_INPUT_SIZE, CONTEXTGATES_OUTPUT_SIZE, new ActivationTanh(), 4);

        infos = Matrix.newRowMatrix(INFOS_DIMENSION);
        context = Matrix.newRowMatrix(CONTEXT_DIMENSION);

        startNewMusic( Matrix.newRowMatrix(0,0,0,0,0) );
    }


    /****************************************************************
     * Copy constructor with an amount of randomization, using the
     * random of a Random object
     * @param cloned The CesureFeedForwardGate object to clone
     * @param rand The Random object
     * @param magnitude The weights random changes magnitude
     ****************************************************************/
    public CesureTest(CesureTest cloned, Random rand, double magnitude) {
        outputGate = new cesure.network.CesureGate(cloned.outputGate, rand, magnitude);
        forgetGate = new cesure.network.CesureGate(cloned.forgetGate, rand, magnitude);
        memoryGate = new cesure.network.CesureGate(cloned.memoryGate, rand, magnitude);
        memoryInputGate  =new cesure.network.CesureGate(cloned.memoryInputGate, rand, magnitude);

        infos = cloned.infos.cp();
        context = cloned.context.cp();
    }



    /****************************************************************
     * Compute the next music's note
     * Will change the context according to the outputed note
     * @return The computed note
     ****************************************************************/
    public Matrix computeNextNote() {
        // Comppute output
        final Matrix infosAndContext = Matrix_concatenateRowMatrix(infos,context);
        final Matrix outputVect = outputGate.compute(infosAndContext);

        // Compute context update
        final Matrix contextInput = Matrix_concatenateRowMatrix(infosAndContext, outputVect);
        final  Matrix forgetVect = forgetGate.compute(contextInput);
        context.pMult(forgetVect);
        final Matrix rememberVect = Matrix_pMult(memoryGate.compute(contextInput), memoryInputGate.compute(contextInput));
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
        final Matrix contextInput = Matrix_concatenateRowMatrix(Matrix_concatenateRowMatrix(infos,context), note);
        final Matrix forgetVect = forgetGate.compute(contextInput);
        context.pMult(forgetVect);
        final Matrix rememberVect = Matrix_pMult(memoryGate.compute(contextInput), memoryInputGate.compute(contextInput));
        context.add(rememberVect);
    }


    /****************************************************************
     * Initialize the network for a new music
     * @param infos The music infos vector for the new music
     ****************************************************************/
    public void startNewMusic(Matrix infos) {
        if (infos.length != INFOS_DIMENSION) {
            throw new NeuralNetworkError("CesureLSTM.startNewMusic(double[])");
        }
        context.setZero();
        for (int i=0; i<infos.length; i++) {
            this.infos.set(i, infos.array[0][i]);
        }
    }

    /****************************************************************
     * Set the gates according to a CesureLSTM object
     * @param network The network to take the gates from
     ****************************************************************/
    public void setGates(CesureTest network) {
        outputGate = network.outputGate;
        forgetGate = network.forgetGate;
        memoryGate = network.memoryGate;
        memoryInputGate = network.memoryInputGate;
        context = network.context;
        infos = network.infos;
    }


    /****************************************************************
     * Calculate the sum of the errors between this output and a
     * CesureMusic object notes
     * @param music The CesureMusic object to compute on
     * @param start The first note to start add errors at :
     *              The "start" first notes' errors won't be added
     *              to the sum
     * @return The error sum
     ****************************************************************/
    public double calculateErrorSum(CesureMusic music, int start) {
        Matrix[] chords = music.chords;
        int nbChords = chords.length;

        startNewMusic(music.infos);
        double errorSum = 0;
        for (int chordI=0; chordI<nbChords; chordI++) {
            if (chordI<start) {
                inputNextNote(chords[chordI]);
            } else {
                errorSum += Math.abs(Matrix_substract(chords[chordI], computeNextNote()).avg());
            }
        }
        return errorSum;
    }




    /****************************************************************
     * Create a new CesureLSTM object from a serialized file
     * @param name The serialized file name
     * @return The newly loaded CesureLSTM object
     ****************************************************************/
    public static CesureTest loadFromSerialization(String name) {
        try {
            return (CesureTest) SerializationManager.load(name+".lstm");
        } catch (IOException e) {
            e.printStackTrace();
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
        }
        throw new NeuralNetworkError("CesureLSTM.loadFromSerialization(String)  : Couldn't load the network");
    }


    /****************************************************************
     * Write a serialized file from a CesureLSTM object
     * @param name The serialized file name
     ****************************************************************/
    public void saveSerialization(String name) {
        try {
            SerializationManager.save("network/save/"+name+".lstm", this);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }



    /****************************************************************
     * Return the total number of neurons in this network
     * @return The total number of neurons in this network
     ****************************************************************/
    public int getNbNeurons() {
        int count = 0;
        count += outputGate.getNbNeurons();
        count += forgetGate.getNbNeurons();
        count += memoryGate.getNbNeurons();
        count += memoryInputGate.getNbNeurons();
        return count;
    }


    /****************************************************************
     * Print the layer
     * @param name The serialized file name
     ****************************************************************/
    public void print(String name) {
        System.out.println("::: Printing "+name+" :::");
        System.out.println("- Output Gate");
        outputGate.print();
        System.out.println("- Forget Gate");
        forgetGate.print();
        System.out.println("- Input Gate");
        memoryGate.print();
        System.out.println("- Remember Gate");
        memoryInputGate.print();
    }

}
