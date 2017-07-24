package cesure.network;

public class CesureComputeInfos {

    public CesureGateComputeInfos outputGateInfos;
    public CesureGateComputeInfos forgetGateInfos;
    public CesureGateComputeInfos memoryGateInfos;
    public CesureGateComputeInfos memoryInputGateInfos;

    public CesureComputeInfos(CesureGateComputeInfos outputGateInfos, CesureGateComputeInfos forgetGateInfos,
                              CesureGateComputeInfos memoryGateInfos, CesureGateComputeInfos memoryInputGateInfos) {
        this.outputGateInfos = outputGateInfos;
        this.forgetGateInfos = forgetGateInfos;
        this.memoryGateInfos = memoryGateInfos;
        this.memoryInputGateInfos = memoryInputGateInfos;
    }

}
