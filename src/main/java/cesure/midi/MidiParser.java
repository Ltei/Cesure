package cesure.midi;

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

import java.io.Serializable;
import java.util.ArrayList;

import static cesure.network.Cesure.*;

public class MidiParser implements Serializable {

    public static final double TEMPO_BPM_RANGE = 1000;

    public static final double TSIGNATURE_NUMERATOR_RANGE = 8;
    public static final double TSIGNATURE_DENOMINATOR_RANGE = 8;
    public static final double TSIGNATURE_METER_RANGE = 96;
    public static final double TSIGNATURE_DIVISION_RANGE = 32;



    /****************************************************************
     * Convert a MidiFile object to a CesureMusic object
     * @param file The MidiFile to convert
     * @return The newly created CesureMusic object
     ****************************************************************/
    public static CesureMusic midiFileToCesureMusic(MidiFile file) {
        Matrix infos = midiToNetworkInfos(file);
        int bpm = (int) Math.round(infos.array[0][0] * TEMPO_BPM_RANGE);
        Matrix[] notes = midiToNetworkChords(file, bpm);
        return new CesureMusic(infos,notes);
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

        double[] data = new double[INFOS_DIMENSION];

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

        // Get first and last octave
        int firstOctave = 100;
        int lastOctave = 0;
        for (MidiTrack track : file.getTracks()) {
            ArrayList<MidiEvent> trackEvents = track.getEventsAsArrayList();
            for (int eventI=0; eventI<trackEvents.size(); eventI++) {
                MidiEvent event = trackEvents.get(eventI);
                if (event instanceof NoteOn) {
                    int octave = ((NoteOn) event).getNoteValue() / 12;
                    if (octave > lastOctave) {lastOctave = octave;}
                    if (octave < firstOctave) {firstOctave = octave;}
                } else if (event instanceof  NoteOff) {
                    int octave = ((NoteOff) event).getNoteValue() / 12;
                    if (octave > lastOctave) {lastOctave = octave;}
                    if (octave < firstOctave) {firstOctave = octave;}
                }
            }
        }
        if (lastOctave-firstOctave+1 > NB_OCTAVES) {
            System.out.println("MidiParser.midiToNetworkNotes(MidiFile,int) : Octave range is too high -> will have to change some notes");
        }

        // Initialize
        double[][] chordsData = new double[(int)(file.getLengthInTicks()/bpm)+1][CHORD_DIMENSION];
        for (int i=0; i<chordsData.length; i++) {
            for (int j=0; j<CHORD_DIMENSION; j++) {
                chordsData[i][j] = 0;
            }
        }

        // Iterate over the tracks' events
        for (MidiTrack track : file.getTracks()) {
            ArrayList<MidiEvent> trackEvents = track.getEventsAsArrayList();

            for (int eventI=0; eventI<trackEvents.size(); eventI++) {
                MidiEvent event = trackEvents.get(eventI);
                if (event instanceof NoteOn) {
                    // Getting note infos
                    NoteOn noteOnEvent = (NoteOn) event;
                    int onTick = (int) noteOnEvent.getTick() / bpm;
                    int noteKey = noteOnEvent.getNoteValue();

                    // Normalize key
                    int normalizedKey = noteKey;
                    while (normalizedKey > lastOctave*12) {
                        normalizedKey -= 12;
                    }
                    normalizedKey -= firstOctave*12;

                    // Looking for the corresponding NoteOff event
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
                        System.out.println("Didn't find note off event from note on at tick #"+onTick);
                    } else {
                        for (int tick=onTick; tick<=offTick; tick++) {
                            chordsData[tick][normalizedKey] = 1;
                        }
                    }
                }
            }
        }

        // Normalization
        Matrix[] chords = new Matrix[chordsData.length];
        for (int i=0; i<chordsData.length; i++) {
            chords[i] = Matrix.newRowMatrix(chordsData[i]);
        }
        return chords;
    }



    /****************************************************************
     * Convert CesureNetowork representation of a midi file to a
     * MidiFile object
     * @param music The CesureMusic object to convert
     * @return The new MidiFile object
     ****************************************************************/
    public static MidiFile cesureMusicToMidiFile(CesureMusic music, int firstOctave) {
        final int firstKey = 12 * firstOctave;

        Matrix infos = music.infos;
        Matrix[] notes = music.chords;

        int bpm = (int) Math.round(infos.array[0][0] * TEMPO_BPM_RANGE);

        MidiTrack headerTrack = new MidiTrack();
        Tempo tempo = new Tempo();
        tempo.setBpm(bpm);
        TimeSignature timeSignature = new TimeSignature(0,0,
                (int) Math.round(infos.array[0][1] * TSIGNATURE_NUMERATOR_RANGE),
                (int) Math.round(infos.array[0][2] * TSIGNATURE_DENOMINATOR_RANGE),
                (int) Math.round(infos.array[0][3] * TSIGNATURE_METER_RANGE),
                (int) Math.round(infos.array[0][4] * TSIGNATURE_DIVISION_RANGE));

        headerTrack.insertEvent(tempo);
        headerTrack.insertEvent(timeSignature);

        MidiTrack noteTrack = new MidiTrack();
        for (int keyI=0; keyI<notes[0].nbColumns; keyI++) {
            boolean noteIsOn = false;
            for (int tickI=0; tickI<notes.length; tickI++){
                if (noteIsOn) {
                    if (notes[tickI].array[0][keyI] <= 0.5) {
                        noteTrack.insertEvent(new NoteOff((tickI-1)*bpm,0, keyI+firstKey, 128));
                        noteIsOn = false;
                    }
                } else {
                    if (notes[tickI].array[0][keyI] >= 0.5) {
                        noteTrack.insertEvent(new NoteOn(tickI*bpm,0, keyI+firstKey, 128));
                        noteIsOn = true;
                    }
                }
            }
            if (noteIsOn) {
                noteTrack.insertEvent(new NoteOff(bpm*(notes.length-1),0, keyI+firstKey, 128));
            }
        }

        ArrayList<MidiTrack> tracks = new ArrayList<MidiTrack>();
        tracks.add(headerTrack);
        tracks.add(noteTrack);

        return new MidiFile(MidiFile.DEFAULT_RESOLUTION, tracks);
    }

}