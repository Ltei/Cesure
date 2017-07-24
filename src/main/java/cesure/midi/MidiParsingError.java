package cesure.midi;

public class MidiParsingError extends RuntimeException {
    public MidiParsingError(String message) {
        super(message);
    }
}
