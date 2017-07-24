package cesure.network;

import cesure.midi.MidiParser;
import cesure.utils.Matrix;

import java.io.File;
import java.io.IOException;

import static cesure.network.Cesure.CHORD_DIMENSION;
import static cesure.network.Cesure.INFOS_DIMENSION;

public class CesureMusic {

    public Matrix infos;
    public Matrix[] chords;

    /****************************************************************
     * Default constructor
     * @param infos The music infos vector for the music
     * @param chords The matrix representation of the music chords
     ****************************************************************/
    public CesureMusic(Matrix infos, Matrix[] chords) {
        if (!infos.isRowMatrix() || infos.nbColumns != INFOS_DIMENSION) {
            throw new NeuralNetworkError("CesureMusic(Matrix,Matrix[])");
        }
        for (Matrix chord : chords) {
            if (!chord.isRowMatrix() || chord.nbColumns != CHORD_DIMENSION) {
                throw new NeuralNetworkError("CesureMusic(Matrix,Matrix[]) : "+chord.nbRows+" - "+chord.nbColumns+" - "+CHORD_DIMENSION);
            }
        }
        this.infos = infos;
        this.chords = chords;
    }


    /****************************************************************
     * Normalize the chords so it can be passed to a CesureMusic object
     * This normalization will just put at 1 the most probable note,
     * and at 0 the others
     ****************************************************************/
    public void normalize_OneNote() {
        for (Matrix chord : chords) {
            int maxOutputI = 0;
            for (int i=1; i<chord.nbColumns; i++) {
                if (chord.array[0][i] > chord.array[0][maxOutputI]) {
                    maxOutputI = i;
                }
            }
            chord.setZero();
            chord.array[0][maxOutputI] = 1;
        }
    }


    /****************************************************************
     * Export to a midi file
     * @param fileName The file name
     ****************************************************************/
    public void exportMidi(String fileName, int firstOctave) {
        try {
            MidiParser.cesureMusicToMidiFile(this, firstOctave).writeToFile(new File("network/midi/"+fileName+".mid"));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /****************************************************************
     * Print all the chords
     ****************************************************************/
    public void print() {
        print(0, chords.length-1);
    }

    /****************************************************************
     * Print the chords from firstChordIndex to lastChordIndex
     ****************************************************************/
    public void print(int firstChordIndex, int lastChordIndex) {
        if (firstChordIndex < 0 || firstChordIndex > lastChordIndex || lastChordIndex < 0 || lastChordIndex >= chords.length) {
            throw new IllegalArgumentException();
        }
        System.out.println("::: PRINTING CESUREMUSIC :::");
        infos.print("- Infos :");
        System.out.println("- Notes :");
        for (int i=firstChordIndex; i<=lastChordIndex; i++) {
            System.out.print(i+" : ");
            chords[i].print();
        }
    }

}
