<?xml version='1.0' encoding='utf-8'?>
<simconf>
  <simulation>
    <title>STACK RPL-UDP-IDS Example DIS Flood Random Attack</title>
    <randomseed>123457</randomseed>
    <motedelay_us>1000000</motedelay_us>
    <radiomedium>
      org.contikios.cooja.radiomediums.UDGM
      <transmitting_range>50.0</transmitting_range>
      <interference_range>100.0</interference_range>
      <success_ratio_tx>1.0</success_ratio_tx>
      <success_ratio_rx>1.0</success_ratio_rx>
    </radiomedium>
    <events>
      <logoutput>40000</logoutput>
      <datatrace>true</datatrace>
    </events>
    <motetype>
      org.contikios.cooja.contikimote.ContikiMoteType
      <identifier>mtype944612407</identifier>
      <description>Cooja Mote Type #1</description>
      <source>/address/to/udp-server.c</source>                                                  // fill it correctly
      <commands>make -j$(CPUS) udp-server.cooja TARGET=cooja</commands>
      <moteinterface>org.contikios.cooja.interfaces.Position</moteinterface>
      <moteinterface>org.contikios.cooja.interfaces.Battery</moteinterface>
      <moteinterface>org.contikios.cooja.contikimote.interfaces.ContikiVib</moteinterface>
      <moteinterface>org.contikios.cooja.contikimote.interfaces.ContikiMoteID</moteinterface>
      <moteinterface>org.contikios.cooja.contikimote.interfaces.ContikiRS232</moteinterface>
      <moteinterface>org.contikios.cooja.contikimote.interfaces.ContikiBeeper</moteinterface>
      <moteinterface>org.contikios.cooja.interfaces.RimeAddress</moteinterface>
      <moteinterface>org.contikios.cooja.contikimote.interfaces.ContikiIPAddress</moteinterface>
      <moteinterface>org.contikios.cooja.contikimote.interfaces.ContikiRadio</moteinterface>
      <moteinterface>org.contikios.cooja.contikimote.interfaces.ContikiButton</moteinterface>
      <moteinterface>org.contikios.cooja.contikimote.interfaces.ContikiPIR</moteinterface>
      <moteinterface>org.contikios.cooja.contikimote.interfaces.ContikiClock</moteinterface>
      <moteinterface>org.contikios.cooja.contikimote.interfaces.ContikiLED</moteinterface>
      <moteinterface>org.contikios.cooja.interfaces.Mote2MoteRelations</moteinterface>
      <moteinterface>org.contikios.cooja.interfaces.MoteAttributes</moteinterface>
    </motetype>
    <motetype>
      org.contikios.cooja.contikimote.ContikiMoteType
      <identifier>mtype900592173</identifier>
      <description>Cooja Mote Type #2</description>
      <source>address/to/udp-client.c</source>                                                   // fill it correctly
      <commands>make -j$(CPUS) udp-client.cooja TARGET=cooja</commands>
      <moteinterface>org.contikios.cooja.interfaces.Position</moteinterface>
      <moteinterface>org.contikios.cooja.interfaces.Battery</moteinterface>
      <moteinterface>org.contikios.cooja.contikimote.interfaces.ContikiVib</moteinterface>
      <moteinterface>org.contikios.cooja.contikimote.interfaces.ContikiMoteID</moteinterface>
      <moteinterface>org.contikios.cooja.contikimote.interfaces.ContikiRS232</moteinterface>
      <moteinterface>org.contikios.cooja.contikimote.interfaces.ContikiBeeper</moteinterface>
      <moteinterface>org.contikios.cooja.interfaces.RimeAddress</moteinterface>
      <moteinterface>org.contikios.cooja.contikimote.interfaces.ContikiIPAddress</moteinterface>
      <moteinterface>org.contikios.cooja.contikimote.interfaces.ContikiRadio</moteinterface>
      <moteinterface>org.contikios.cooja.contikimote.interfaces.ContikiButton</moteinterface>
      <moteinterface>org.contikios.cooja.contikimote.interfaces.ContikiPIR</moteinterface>
      <moteinterface>org.contikios.cooja.contikimote.interfaces.ContikiClock</moteinterface>
      <moteinterface>org.contikios.cooja.contikimote.interfaces.ContikiLED</moteinterface>
      <moteinterface>org.contikios.cooja.interfaces.Mote2MoteRelations</moteinterface>
      <moteinterface>org.contikios.cooja.interfaces.MoteAttributes</moteinterface>
    </motetype>
    <mote>
      <interface_config>
        org.contikios.cooja.interfaces.Position
        <x>43.291897546941804</x>
        <y>7.17470867058031</y>
        <z>0.0</z>
      </interface_config>
      <interface_config>
        org.contikios.cooja.contikimote.interfaces.ContikiMoteID
        <id>1</id>
      </interface_config>
      <motetype_identifier>mtype944612407</motetype_identifier>
    </mote>
    <mote>
      <interface_config>
        org.contikios.cooja.interfaces.Position
        <x>88.51821688873832</x>
        <y>27.70818809039509</y>
        <z>0.0</z>
      </interface_config>
      <interface_config>
        org.contikios.cooja.contikimote.interfaces.ContikiMoteID
        <id>2</id>
      </interface_config>
      <motetype_identifier>mtype900592173</motetype_identifier>
    </mote>
    <mote>
      <interface_config>
        org.contikios.cooja.interfaces.Position
        <x>77.01497869386418</x>
        <y>70.53565733865031</y>
        <z>0.0</z>
      </interface_config>
      <interface_config>
        org.contikios.cooja.contikimote.interfaces.ContikiMoteID
        <id>3</id>
      </interface_config>
      <motetype_identifier>mtype900592173</motetype_identifier>
    </mote>
    <mote>
      <interface_config>
        org.contikios.cooja.interfaces.Position
        <x>26.786746555544028</x>
        <y>21.65982134704491</y>
        <z>0.0</z>
      </interface_config>
      <interface_config>
        org.contikios.cooja.contikimote.interfaces.ContikiMoteID
        <id>4</id>
      </interface_config>
      <interface_config>
        org.contikios.cooja.contikimote.interfaces.ContikiRS232
        <history>asdf~;</history>
      </interface_config>
      <motetype_identifier>mtype900592173</motetype_identifier>
    </mote>
    <mote>
      <interface_config>
        org.contikios.cooja.interfaces.Position
        <x>71.14145737675113</x>
        <y>16.76391167245226</y>
        <z>0.0</z>
      </interface_config>
      <interface_config>
        org.contikios.cooja.contikimote.interfaces.ContikiMoteID
        <id>5</id>
      </interface_config>
      <motetype_identifier>mtype900592173</motetype_identifier>
    </mote>
    <mote>
      <interface_config>
        org.contikios.cooja.interfaces.Position
        <x>60.406859080436845</x>
        <y>62.57546806192654</y>
        <z>0.0</z>
      </interface_config>
      <interface_config>
        org.contikios.cooja.contikimote.interfaces.ContikiMoteID
        <id>6</id>
      </interface_config>
      <motetype_identifier>mtype900592173</motetype_identifier>
    </mote>
    <mote>
      <interface_config>
        org.contikios.cooja.interfaces.Position
        <x>42.35294362915937</x>
        <y>11.229999778982364</y>
        <z>0.0</z>
      </interface_config>
      <interface_config>
        org.contikios.cooja.contikimote.interfaces.ContikiMoteID
        <id>7</id>
      </interface_config>
      <motetype_identifier>mtype900592173</motetype_identifier>
    </mote>
    <mote>
      <interface_config>
        org.contikios.cooja.interfaces.Position
        <x>102.22474191180771</x>
        <y>88.42349556421627</y>
        <z>0.0</z>
      </interface_config>
      <interface_config>
        org.contikios.cooja.contikimote.interfaces.ContikiMoteID
        <id>8</id>
      </interface_config>
      <motetype_identifier>mtype900592173</motetype_identifier>
    </mote>
    <mote>
      <interface_config>
        org.contikios.cooja.interfaces.Position
        <x>53.68903552131917</x>
        <y>99.40475724061578</y>
        <z>0.0</z>
      </interface_config>
      <interface_config>
        org.contikios.cooja.contikimote.interfaces.ContikiMoteID
        <id>9</id>
      </interface_config>
      <motetype_identifier>mtype900592173</motetype_identifier>
    </mote>
    <mote>
      <interface_config>
        org.contikios.cooja.interfaces.Position
        <x>64.68510831121034</x>
        <y>69.5719019941184</y>
        <z>0.0</z>
      </interface_config>
      <interface_config>
        org.contikios.cooja.contikimote.interfaces.ContikiMoteID
        <id>10</id>
      </interface_config>
      <motetype_identifier>mtype900592173</motetype_identifier>
    </mote>
    <mote>
      <interface_config>
        org.contikios.cooja.interfaces.Position
        <x>113.25447997962323</x>
        <y>77.3741619969428</y>
        <z>0.0</z>
      </interface_config>
      <interface_config>
        org.contikios.cooja.contikimote.interfaces.ContikiMoteID
        <id>11</id>
      </interface_config>
      <motetype_identifier>mtype900592173</motetype_identifier>
    </mote>
  </simulation>
  <plugin>
    org.contikios.cooja.plugins.SimControl
    <width>302</width>
    <z>1</z>
    <height>157</height>
    <location_x>310</location_x>
    <location_y>0</location_y>
  </plugin>
  <plugin>
    org.contikios.cooja.plugins.Visualizer
    <plugin_config>
      <moterelations>true</moterelations>
      <skin>org.contikios.cooja.plugins.skins.IDVisualizerSkin</skin>
      <skin>org.contikios.cooja.plugins.skins.UDGMVisualizerSkin</skin>
      <skin>org.contikios.cooja.plugins.skins.MoteTypeVisualizerSkin</skin>
      <viewport>1.68758401899377 0.0 0.0 1.68758401899377 -43.185404471347574 22.822079273955353</viewport>
    </plugin_config>
    <width>306</width>
    <z>0</z>
    <height>328</height>
    <location_x>1</location_x>
    <location_y>0</location_y>
  </plugin>
  <plugin>
    org.contikios.cooja.plugins.LogListener
    <plugin_config>
      <filter>(App|ATTACK)</filter>
      <formatted_time />
      <coloring />
    </plugin_config>
    <width>893</width>
    <z>2</z>
    <height>470</height>
    <location_x>0</location_x>
    <location_y>326</location_y>
  </plugin>
  <plugin>
    org.contikios.cooja.plugins.ScriptRunner
    <plugin_config>
      <script>var senders = {};
var verbose = false;
var waiting_for_stable_network = true;
var sinkId = 1;
// Number of clients (the sink excluded)
var clients = sim.getMotesCount() - 1;
var msgrecv = /.+INFO: App.+Received +message.+ from ([0-9a-f:]+).*/;
var r = new java.util.Random(sim.getRandomSeed());

/* timeout in milliseconds */
TIMEOUT(54000000);

function f(value) {
  return (Math.round(value * 100) / 100).toFixed(2);
}

function setBool(mote, name, value) {
  var mem = new org.contikios.cooja.mote.memory.VarMemory(mote.getMemory());
  if (!mem.variableExists(name)) {
    log.log("ERR: could not find variable '" + name + "'\n");
    return false;
  }
  var symbol = mem.getVariable(name);
  if (verbose) {
    var oldValue = mem.getInt8ValueOf(symbol.addr) ? "true" : "false";
    log.log("Set bool " + name + " (address 0x" + java.lang.Long.toHexString(symbol.addr)
            + "/" + symbol.size + ": " + oldValue + ") to " + value + "\n");
  }
  mem.setInt8ValueOf(symbol.addr, value);
  return true;
}

function setInt16(mote, name, value) {
  var mem = new org.contikios.cooja.mote.memory.VarMemory(mote.getMemory());
  if (!mem.variableExists(name)) {
    log.log("ERR: could not find variable '" + name + "'\n");
    return false;
  }
  var symbol = mem.getVariable(name);
  if (verbose) {
    var oldValue = mem.getInt16ValueOf(symbol.addr) &amp; 0xffff;
    log.log("Set int16 " + name + " (address 0x" + java.lang.Long.toHexString(symbol.addr)
            + "/" + symbol.size + ": " + oldValue + ") to " + value + "\n");
  }
  mem.setInt16ValueOf(symbol.addr, value);
  return true;
}

function selectAttacker() {
  var sink = sim.getMoteWithID(sinkId);
  var sinkRadio = sink.getInterfaces().getRadio();
  while (true) {
    var attackerId = 2 + r.nextInt(clients);
    log.log("Checking potential attacker " + attackerId + "... ");

    var attacker = sim.getMoteWithID(attackerId);
    var attackerRadio = attacker.getInterfaces().getRadio();
    var neighbours = sim.getRadioMedium().getNeighbours(attackerRadio);
    if (neighbours.contains(sinkRadio)) {
      log.log("[FAIL] - has sink as neighbour\n");
      continue;
    }
    log.log("[OK]\n");
    return attacker;
  }
}

while(waiting_for_stable_network) {
    YIELD();
    if (id == 1) {
        match = msg.match(msgrecv)
        if (match) {
            senders[match[1]] = true;
            var size = Object.keys(senders).length;
            log.log("Sink has contact with " + match[1] + " (" + (clients - size) + " remaining)\n");
            if (size &gt;= clients) {
                log.log("Sink has contact with all clients!\n");
                waiting_for_stable_network = false;
            }
        }
    }
}

GENERATE_MSG(2000, "continue");
YIELD_THEN_WAIT_UNTIL(msg.equals("continue"));

sim.getEventCentral().logEvent("network", "steady-state");
log.log("Network steady state!\n");

GENERATE_MSG(26000000, "continue");
YIELD_THEN_WAIT_UNTIL(msg.equals("continue"));

var attacker = selectAttacker();
log.log("Network DIS flooding attack from " + attacker.getID() + "!\n");
sim.getEventCentral().logEvent("attack", "dfa:" + attacker.getID());

/* Configure DIS flood random attack */
setBool(attacker, 'network_attacks_rpl_dfa', true);
setInt16(attacker, 'network_attacks_rpl_dfa_period', 1000);

GENERATE_MSG(26000000, "continue");
YIELD_THEN_WAIT_UNTIL(msg.equals("continue"));

log.testOK();</script>
      <active>true</active>
    </plugin_config>
    <width>600</width>
    <z>3</z>
    <height>642</height>
    <location_x>900</location_x>
    <location_y>0</location_y>
  </plugin>
</simconf>