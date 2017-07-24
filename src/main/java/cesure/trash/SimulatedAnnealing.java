package cesure.trash;

import cesure.network.CesureMusic;

class SimulatedAnnealing {

    /****************************************************************
     * The getSimulatedAnnealingOutput output
     * Containing the best network and its error
     ****************************************************************/
    public static class SimulatedAnnealingOutput {
        private CesureLSTM network;
        private double error_sum;
        private SimulatedAnnealingOutput(CesureLSTM network, double error_sum) {
            this.network = network;
            this.error_sum = error_sum;
        }
    }


    /****************************************************************
     * Perform a simulated annealing training on one music
     * @param network The CesureLSTM object to perform on
     * @param music The CesureMusic object to train on
     * @param start The first note to start guess at :
     *              The "start" first notes will change the context
     *              without being trained on
     * @param magnitude The magnitude
     * @param iterations The number of iterations
     ****************************************************************/
    public static void train(CesureLSTM network, CesureMusic music, int start, double magnitude, int iterations) {
        double actualMagnitude;
        SimulatedAnnealingOutput output = getSimulatedAnnealingOutput(network,music,start,magnitude);
        System.out.println("Epoch #0 - Error = "+output.error_sum);
        for (int epochI=1; epochI<iterations; epochI++) {
            actualMagnitude = (iterations-epochI) * magnitude / iterations;
            output = getSimulatedAnnealingOutput(output,music,start,actualMagnitude);
            System.out.println("Epoch #"+epochI+" - Error = "+output.error_sum);
        }
        network.setGates(output.network);
    }


    /****************************************************************
     * Return the best network between network and newNetwork,
     * with it's error sum, wrapped in a SimulatedAnnealingOutput
     * object
     * @param network The network to perform simulated annealing on
     * @param music The music to compute error on
     * @param start The first note to start add errors at :
     *              The "start" first notes' errors won't be added
     *              to the sum
     * @param magnitude The magnitude
     * @return A SimulatedAnnealingOutput object
     ****************************************************************/
    static SimulatedAnnealingOutput getSimulatedAnnealingOutput(CesureLSTM network, CesureMusic music, int start, double magnitude) {
        CesureLSTM newNetwork = new CesureLSTM(network, magnitude);
        double oldError_sum = network.calculateErrorSum(music,start);
        double newError_sum = newNetwork.calculateErrorSum(music,start);
        if (newError_sum < oldError_sum) {
            return new SimulatedAnnealingOutput(newNetwork,newError_sum);
        } else {
            return new SimulatedAnnealingOutput(network, oldError_sum);
        }
    }
    /****************************************************************
     * Return the best network between network and newNetwork,
     * with it's error sum, wrapped in a SimulatedAnnealingOutput
     * object
     * Won't compute network's error, since it's in the parameters
     * @param output The previous simulated annealing output
     * @param music The music to compute error on
     * @param start The first note to start add errors at :
     *              The "start" first notes' errors won't be added
     *              to the sum
     * @param magnitude The magnitude
     * @return A SimulatedAnnealingOutput object
     ****************************************************************/
    static SimulatedAnnealingOutput getSimulatedAnnealingOutput(SimulatedAnnealingOutput output, CesureMusic music, int start, double magnitude) {
        CesureLSTM newNetwork = new CesureLSTM(output.network, magnitude);
        double newError_sum = newNetwork.calculateErrorSum(music,start);
        if (newError_sum < output.error_sum) {
            return new SimulatedAnnealingOutput(newNetwork,newError_sum);
        } else {
            return output;
        }
    }

}
