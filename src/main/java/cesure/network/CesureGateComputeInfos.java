package cesure.network;

import cesure.utils.Matrix;

public class CesureGateComputeInfos {

    public final Matrix[] hiddens_unact;
    public final Matrix[] hiddens;
    public final Matrix output_unact;
    public final Matrix output;

    public CesureGateComputeInfos(Matrix[] hiddens_unact, Matrix[] hiddens, Matrix output_unact, Matrix output) {
        this.hiddens_unact = hiddens_unact;
        this.hiddens = hiddens;
        this.output_unact = output_unact;
        this.output = output;
    }

}
