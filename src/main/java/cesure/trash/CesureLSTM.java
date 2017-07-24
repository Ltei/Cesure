package cesure.trash;

import cesure.midi.MidiParser;
import cesure.network.CesureMusic;
import cesure.network.NeuralNetworkError;
import cesure.network.activation.ActivationSigmoid;
import cesure.network.activation.ActivationTanh;
import cesure.utils.Matrix;
import cesure.utils.SerializationManager;

import java.io.*;
import java.util.Random;

import static cesure.utils.MatrixMath.Matrix_concatenateRowMatrix;
import static cesure.utils.MatrixMath.Matrix_pMult;
import static cesure.utils.MatrixMath.Matrix_substract;

class CesureLSTM implements Serializable {

    private final int outputDimension;

    private final int infosDimension;
    private final int contextDimension;

    private final int noteInputDimension;
    private final int contextInputDimension;

    private CesureGate outputGate;
    private CesureGate forgetGate;
    private CesureGate inputGate;
    private CesureGate rememberGate;

    private Matrix context;
    private Matrix infos;


    /****************************************************************
     * Default constructor
     ****************************************************************/
    public CesureLSTM() {
        outputDimension = MidiParser.NOTE_DATA_SIZE;
        infosDimension = MidiParser.INFO_DATA_SIZE;

        contextDimension = 150;

        noteInputDimension = infosDimension + contextDimension;
        contextInputDimension = noteInputDimension + outputDimension;

        infos = Matrix.newRowMatrix(infosDimension);
        context = Matrix.newRowMatrix(contextDimension);

        startNewMusic( Matrix.newRowMatrix(new double[] {0,0,0,0,0}) );

        outputGate = new CesureGate(noteInputDimension, outputDimension, 4,75, new ActivationSigmoid());
        forgetGate = new CesureGate(contextInputDimension, contextDimension, 4,75, new ActivationSigmoid());
        inputGate = new CesureGate(contextInputDimension, contextDimension, 4,75, new ActivationSigmoid());
        rememberGate = new CesureGate(contextInputDimension, contextDimension, 4,75, new ActivationTanh());
    }


    /****************************************************************
     * Copy constructor with random changes
     * @param toClone The CesureLSTM object to clone
     * @param randomMagnitude The random changes magnitude
     ****************************************************************/
    public CesureLSTM(CesureLSTM toClone, double randomMagnitude) {
        outputDimension = toClone.outputDimension;

        infosDimension = toClone.infosDimension;
        contextDimension = toClone.contextDimension;

        noteInputDimension = toClone.noteInputDimension;
        contextInputDimension = toClone.contextInputDimension;

        infos = toClone.infos.cp();
        context = toClone.context.cp();

        outputGate = new CesureGate(toClone.outputGate, randomMagnitude);
        forgetGate = new CesureGate(toClone.forgetGate, randomMagnitude);
        inputGate = new CesureGate(toClone.inputGate, randomMagnitude);
        rememberGate = new CesureGate(toClone.rememberGate, randomMagnitude);
    }


    public CesureLSTM(CesureLSTM toClone, Random rand, double randomMagnitude) {
        outputDimension = toClone.outputDimension;

        infosDimension = toClone.infosDimension;
        contextDimension = toClone.contextDimension;

        noteInputDimension = toClone.noteInputDimension;
        contextInputDimension = toClone.contextInputDimension;

        infos = toClone.infos.cp();
        context = toClone.context.cp();

        outputGate = new CesureGate(toClone.outputGate, rand, randomMagnitude);
        forgetGate = new CesureGate(toClone.forgetGate, rand, randomMagnitude);
        inputGate = new CesureGate(toClone.inputGate, rand, randomMagnitude);
        rememberGate = new CesureGate(toClone.rememberGate, rand, randomMagnitude);
    }


    /****************************************************************
     * Compute the next music's note
     * Will change the context according to the outputed note
     * @return The computed note
     ****************************************************************/
    public Matrix computeNextNote() {
        // Comppute output
        final Matrix noteInput = Matrix_concatenateRowMatrix(infos,context);
        final Matrix outputVect = outputGate.compute(noteInput);

        // Normalize output : Take only the most probable note
        /*final Matrix normalizedOutput = outputVect.cp();
        int maxOutputI = 0;
        for (int i=1; i<normalizedOutput.nbColumns; i++) {
            if (normalizedOutput.array[0][i] > normalizedOutput.array[0][maxOutputI]) {
                maxOutputI = i;
            }
        }
        outputVect.setZero();
        outputVect.array[0][maxOutputI] = 1;*/

        // Compute context update
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
        final Matrix contextInput = Matrix_concatenateRowMatrix(Matrix_concatenateRowMatrix(infos,context), note);
        final Matrix forgetVect = forgetGate.compute(contextInput);
        context.pMult(forgetVect);
        final Matrix rememberVect = Matrix_pMult(inputGate.compute(contextInput), rememberGate.compute(contextInput));
        context.add(rememberVect);
    }


    /****************************************************************
     * Initialize the network for a new music
     * @param infos The music infos vector for the new music
     ****************************************************************/
    public void startNewMusic(Matrix infos) {
        if (infos.length != infosDimension) {
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
    public void setGates(CesureLSTM network) {
        outputGate = network.outputGate;
        forgetGate = network.forgetGate;
        inputGate = network.inputGate;
        rememberGate = network.rememberGate;
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
        Matrix[] notes = music.notes;
        int nbNotes = notes.length;

        startNewMusic(music.infos);
        double errorSum = 0;
        for (int noteI=0; noteI<nbNotes; noteI++) {
            if (noteI<start) {
                inputNextNote(notes[noteI]);
            } else {
                errorSum += Math.abs(Matrix_substract(notes[noteI], computeNextNote()).avg());
            }
        }
        return errorSum;
    }


    /****************************************************************
     * Calculate the number of times the network's most probable
     * output isn't one of the music's note
     * @param music The CesureMusic object to compute on
     * @param start The first note to start add errors at :
     *              The "start" first notes' errors won't be added
     *              to the sum
     * @return The error sum
     ****************************************************************/
    /*public double calculateErrorSum(CesureMusic music, int start) {
        Matrix[] notes = music.notes;
        int nbNotes = notes.length;

        startNewMusic(music.infos);
        double errorSum = 0;
        for (int noteI=0; noteI<nbNotes; noteI++) {
            if (noteI<start) {
                inputNextNote(notes[noteI]);
            } else {
                Matrix output = computeNextNote();
                // If the most probable note in the output isn't on in the music at this moment,
                // errorSum += 1
                int maxOutputI = 0;
                for (int i=1; i<output.nbColumns; i++) {
                    if (output.array[0][i] > output.array[0][maxOutputI]) {
                        maxOutputI = i;
                    }
                }
                if (notes[noteI].array[0][maxOutputI] != 1) {
                    errorSum += 1;
                }
            }
        }
        return errorSum;
    }*/




    /****************************************************************
     * Create a new CesureLSTM object from a serialized file
     * @param name The serialized file name
     * @return The newly loaded CesureLSTM object
     ****************************************************************/
    public static CesureLSTM loadFromSerialization(String name) {
        try {
            return (CesureLSTM) SerializationManager.load(name+".lstm");
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
        count += inputGate.getNbNeurons();
        count += rememberGate.getNbNeurons();
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
        inputGate.print();
        System.out.println("- Remember Gate");
        rememberGate.print();
    }

}
