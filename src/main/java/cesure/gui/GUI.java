package cesure.gui;

import javax.swing.*;
import java.awt.*;

public class GUI {

    public static final String ACTION_NEWNETWORK = "ACTION_NEWNETWORK";

    public GUI() {
        JFrame frame = new JFrame("Cesure");
        frame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        frame.setBackground(Color.cyan);
        Dimension size = new Dimension(500,800);
        frame.setPreferredSize(size);
        frame.setSize(size);
        frame.setLocationRelativeTo(null);

        JPanel panel = new GUIPanel();
        panel.setBackground(Color.cyan);
        panel.setOpaque(true); //content panes must be opaque
        frame.setContentPane(panel);

        frame.pack();
        frame.setVisible(true);
    }
}
