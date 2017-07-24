package cesure.gui;

import javax.swing.*;
import java.awt.event.ActionListener;
import java.awt.event.KeyEvent;

import static cesure.gui.GUI.ACTION_NEWNETWORK;

class GUIPanel extends JPanel {

    public JButton buttonCreateNetwork;

    public JButton button0;
    public JButton button1;
    public JButton button2;

    GUIPanel() {
        ActionListener listener = new GUIListener(this);

        button0 = new JButton("Disable middle button");
        button0.setVerticalTextPosition(AbstractButton.CENTER);
        button0.setHorizontalTextPosition(AbstractButton.LEADING); //aka LEFT, for left-to-right locales
        button0.setMnemonic(KeyEvent.VK_D);
        button0.setActionCommand("disable");

        button1 = new JButton("Middle button");
        button1.setVerticalTextPosition(AbstractButton.BOTTOM);
        button1.setHorizontalTextPosition(AbstractButton.CENTER);
        button1.setMnemonic(KeyEvent.VK_M);

        button2 = new JButton("Enable middle button");
        //Use the default text position of CENTER, TRAILING (RIGHT).
        button2.setMnemonic(KeyEvent.VK_E);
        button2.setActionCommand("enable");
        button2.setEnabled(false);

        //Listen for actions on buttons 1 and 3.
        button0.addActionListener(listener);
        button2.addActionListener(listener);

        button0.setToolTipText("Click this button to disable the middle button.");
        button1.setToolTipText("This middle button does nothing when you click it.");
        button2.setToolTipText("Click this button to enable the middle button.");

        //Add Components to this container, using the default FlowLayout.
        add(button0);
        add(button1);
        add(button2);

        buttonCreateNetwork = new JButton("New network");
        /*buttonCreateNetwork.setVerticalTextPosition(AbstractButton.BOTTOM);
        buttonCreateNetwork.setHorizontalTextPosition(AbstractButton.CENTER);*/
        buttonCreateNetwork.setBounds(100, 100, 50, 50);
        buttonCreateNetwork.setMnemonic(KeyEvent.VK_M);
        buttonCreateNetwork.setActionCommand(ACTION_NEWNETWORK);
        buttonCreateNetwork.addActionListener(listener);
        buttonCreateNetwork.setToolTipText("Create a new not-trained neural network");

        add(buttonCreateNetwork);
    }

}
