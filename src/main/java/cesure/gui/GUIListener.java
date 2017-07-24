package cesure.gui;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

class GUIListener implements ActionListener {

    private GUIPanel panel;

    GUIListener(GUIPanel panel) {
        this.panel = panel;
    }

    public void actionPerformed(ActionEvent e) {

        if ("disable".equals(e.getActionCommand())) {
            panel.button1.setEnabled(false);
            panel.button0.setEnabled(false);
            panel.button2.setEnabled(true);
        } else {
            panel.button1.setEnabled(true);
            panel.button0.setEnabled(true);
            panel.button2.setEnabled(false);
        }
    }


}
