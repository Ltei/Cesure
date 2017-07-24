package cesure.trash;

import cesure.network.training.NetworkAndError;
import cesure.network.CesureMusic;

class GeneticAnnealing {

    private static class FamilyCreator extends Thread {

        private final NetworkAndError parent;
        private final int generations;

        private final CesureMusic music;
        private final int start;
        private final double magnitude;

        private NetworkAndError output;

        /****************************************************************
         * Constructor
         * @param parent The first parent
         * @param generations The number of generations (one child per
         *                    generation)
         * @param music The music to compute network's error on
         * @param start See SimulatedAnnealing.java
         * @param magnitude The magnitude for each child
         ****************************************************************/
        private FamilyCreator(NetworkAndError parent, int generations, CesureMusic music, int start, double magnitude) {
            this.parent = parent;
            this.generations = generations;
            this.music = music;
            this.start = start;
            this.magnitude = magnitude;
        }

        @Override
        public void run() {
            // Initialize
            final NetworkAndError[] childs = new NetworkAndError[generations];
            // Generate first child
            CesureLSTM firstChild = new CesureLSTM(parent.network,magnitude);
            childs[0] = new NetworkAndError(firstChild, firstChild.calculateErrorSum(music,start));
            // Generate next childs
            for (int genI=1; genI<generations; genI++) {
                final CesureLSTM child = new CesureLSTM(childs[genI-1].network, magnitude);
                final double childError = child.calculateErrorSum(music,start);
                childs[genI] = new NetworkAndError(child,childError);
            }
            // Select best child
            output = parent;
            for (NetworkAndError network : childs) {
                if (network.error_sum < output.error_sum) {
                    output = network;
                }
            }
        }
    }


    /****************************************************************
     * Perform a genetic annealing training on one music
     * Use multiThreading : each processors will compute one family
     * of generationsPerFamily generations, out of the first
     * parent (the generation will only be one child while the others
     * will be childPerParent)
     * @param music The CesureMusic object to train on
     * @param start The first note to start guess at :
     *              The "start" first notes will change the context
     *              without being trained on
     * @param magnitude The magnitude
     * @param iterations The number of iterations
     ****************************************************************/
    public static void train(CesureLSTM network, CesureMusic music, int start, double magnitude, int iterations) {

        final int nbProcessors = Runtime.getRuntime().availableProcessors();

        final int generationsPerFamily = 5;

        NetworkAndError bestNetwork = new NetworkAndError(network, network.calculateErrorSum(music,start));

        for (int epochI=0; epochI<iterations; epochI++) {
            final double actualMagnitude = (iterations-epochI) * magnitude / iterations;

            final FamilyCreator[] familyCreators = new FamilyCreator[nbProcessors];

            for (int threadI=0; threadI<nbProcessors; threadI++) {
                familyCreators[threadI] = new FamilyCreator(bestNetwork, generationsPerFamily, music, start, actualMagnitude);
                familyCreators[threadI].start();
            }
            try {
                for (int threadI=0; threadI<nbProcessors; threadI++) {
                    familyCreators[threadI].join();
                }
            } catch (InterruptedException e) {
                e.printStackTrace();
            }


            for (int threadI=0; threadI<nbProcessors; threadI++) {
                if (familyCreators[threadI].output.error_sum < bestNetwork.error_sum) {
                    bestNetwork = familyCreators[threadI].output;
                }
            }

            System.out.println("Epoch #"+epochI+" - Error = "+bestNetwork.error_sum);
        }
    }

}
