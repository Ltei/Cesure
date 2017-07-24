package cesure.trash;

import cesure.midi.MidiParser;
import cesure.network.CesureMusic;
import cesure.network.NeuralNetworkError;
import cesure.utils.Matrix;
import com.leff.midi.MidiFile;
import com.leff.midi.MidiTrack;
import com.leff.midi.event.MidiEvent;
import com.leff.midi.event.NoteOff;
import com.leff.midi.event.NoteOn;
import com.leff.midi.event.meta.Tempo;
import com.leff.midi.event.meta.TimeSignature;

import java.util.ArrayList;

import static cesure.network.Cesure.CHORD_DIMENSION;
import static cesure.network.Cesure.INFOS_DIMENSION;

class MidiParser2 {

    public static final double TEMPO_BPM_RANGE = 1000;
    public static final double TSIGNATURE_NUMERATOR_RANGE = 8;
    public static final double TSIGNATURE_DENOMINATOR_RANGE = 8;
    public static final double TSIGNATURE_METER_RANGE = 96;
    public static final double TSIGNATURE_DIVISION_RANGE = 32;

    public static final double OCTAVE_RANGE = 10;





    /****************************************************************
     * Convert a MidiFile object to a CesureMusic object
     * @param file The MidiFile to convert
     * @return The newly created CesureMusic object
     ****************************************************************/
    public static CesureMusic midiFileToCesureMusic(MidiFile file) {
        Matrix infos = midiToNetworkInfos(file);
        int bpm = (int) Math.round(infos.array[0][0] * TEMPO_BPM_RANGE);
        Matrix[] chords = midiToNetworkChords(file, bpm);
        return new CesureMusic(infos,chords);
    }


    /****************************************************************
     * Convert a MidiFile object to a Matrix, representing the network
     * music infos
     * @param file The file to convert
     * @return The network infos
     ****************************************************************/
    private static Matrix midiToNetworkInfos(MidiFile file) {

        boolean foundTempo = false;
        boolean foundTimeSignature = false;

        double[] data = new double[INFOS_DIMENSION /*=5*/];

        for (MidiEvent event : file.getTracks().get(0).getEvents()) {
            if (event instanceof Tempo) {
                Tempo tempo = (Tempo) event;
                data[0] = tempo.getBpm() / TEMPO_BPM_RANGE;
                foundTempo = true;
            } else if (event instanceof TimeSignature) {
                TimeSignature timeSignature = (TimeSignature) event;
                data[1] = timeSignature.getNumerator() / TSIGNATURE_NUMERATOR_RANGE;
                data[2] = timeSignature.getDenominatorValue() / TSIGNATURE_DENOMINATOR_RANGE;
                data[3] = timeSignature.getMeter() / TSIGNATURE_METER_RANGE;
                data[4] = timeSignature.getDivision() / TSIGNATURE_DIVISION_RANGE;
                foundTimeSignature = true;
            }
        }
        if (!foundTempo) {
            throw new NeuralNetworkError(MidiParser.class.getSimpleName()+".midiToNetworkInfos(MidiFile) - Didn't find tempo");
        }
        if (!foundTimeSignature) {
            throw new NeuralNetworkError(MidiParser.class.getSimpleName()+".midiToNetworkInfos(MidiFile) - Didn't find time signature");
        }
        return Matrix.newRowMatrix(data);
    }


    /****************************************************************
     * Convert a MidiFile object to an array of Matrix, representing
     * network output chords
     * @param file The file to normalize as network chords
     * @return The network chords array
     ****************************************************************/
    private static Matrix[] midiToNetworkChords(MidiFile file, int bpm) {

        double[][] chordsData = new double[(int)(file.getLengthInTicks()/bpm)+1][CHORD_DIMENSION];
        for (int i=0; i<chordsData.length; i++) {
            for (int j=0; j<CHORD_DIMENSION; j++) {
                chordsData[i][j] = 0;
            }
        }

        for (MidiTrack track : file.getTracks()) {
            // Put events in an Array
            ArrayList<MidiEvent> trackEvents = track.getEventsAsArrayList();
            // Iterate over the events
            for (int eventI=0; eventI<trackEvents.size(); eventI++) {

                MidiEvent event = trackEvents.get(eventI);
                if (event instanceof NoteOn) {
                    NoteOn noteOnEvent = (NoteOn) event;
                    int onTick = (int) noteOnEvent.getTick() / bpm;

                    int noteKey = noteOnEvent.getNoteValue();
                    int noteI = noteKey % 12;
                    int noteOctave = noteKey / 12;

                    int offTick = -1;
                    boolean foundNoteOff = false;
                    for (int offEventI=eventI+1; offEventI<trackEvents.size() && !foundNoteOff; offEventI++) {
                        if (trackEvents.get(offEventI) instanceof NoteOff) {
                            NoteOff noteOffEvent = (NoteOff) trackEvents.get(offEventI);
                            if (noteKey == noteOffEvent.getNoteValue()) {
                                offTick = (int) noteOffEvent.getTick() / bpm;
                                foundNoteOff = true;
                            }
                        }
                    }

                    if (!foundNoteOff) {
                        throw new NeuralNetworkError("MidiParser2.midiToNetworkChords(MidiFile,int) : Didn't find note off event from note on at tick #"+onTick);
                    } else {
                        for (int tick=onTick; tick<=offTick; tick++) {
                            if (chordsData[tick][noteI*2] == 1) {
                                System.out.println("MidiParser2.midiToNetworkChords(MidiFile,int) : Note collision at tick #"+tick);
                            } else {
                                chordsData[tick][noteI*2] = 1; // proba
                                chordsData[tick][noteI*2+1] = noteOctave / OCTAVE_RANGE; // octave
                            }
                        }
                    }
                }

            }
        }

        Matrix[] output = new Matrix[chordsData.length];
        for (int i=0; i<chordsData.length; i++) {
            output[i] = Matrix.newRowMatrix(chordsData[i]);
        }

        return output;
    }

}
