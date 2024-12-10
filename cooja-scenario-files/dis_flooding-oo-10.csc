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
      <identifier>mtype269414323</identifier>
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
      <identifier>mtype547873133</identifier>
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
      <motetype_identifier>mtype269414323</motetype_identifier>
    </mote>
    <mote>
      <interface_config>
        org.contikios.cooja.interfaces.Position
        <x>76.46884404727058</x>
        <y>18.009785387028103</y>
        <z>0.0</z>
      </interface_config>
      <interface_config>
        org.contikios.cooja.contikimote.interfaces.ContikiMoteID
        <id>2</id>
      </interface_config>
      <motetype_identifier>mtype547873133</motetype_identifier>
    </mote>
    <mote>
      <interface_config>
        org.contikios.cooja.interfaces.Position
        <x>70.13884255239748</x>
        <y>35.733155540380714</y>
        <z>0.0</z>
      </interface_config>
      <interface_config>
        org.contikios.cooja.contikimote.interfaces.ContikiMoteID
        <id>3</id>
      </interface_config>
      <motetype_identifier>mtype547873133</motetype_identifier>
    </mote>
    <mote>
      <interface_config>
        org.contikios.cooja.interfaces.Position
        <x>16.824810451248766</x>
        <y>27.64161133807022</y>
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
      <motetype_identifier>mtype547873133</motetype_identifier>
    </mote>
    <mote>
      <interface_config>
        org.contikios.cooja.interfaces.Position
        <x>112.04748126376461</x>
        <y>58.66017771711428</y>
        <z>0.0</z>
      </interface_config>
      <interface_config>
        org.contikios.cooja.contikimote.interfaces.ContikiMoteID
        <id>5</id>
      </interface_config>
      <motetype_identifier>mtype547873133</motetype_identifier>
    </mote>
    <mote>
      <interface_config>
        org.contikios.cooja.interfaces.Position
        <x>148.66967058236833</x>
        <y>66.97996290911944</y>
        <z>0.0</z>
      </interface_config>
      <interface_config>
        org.contikios.cooja.contikimote.interfaces.ContikiMoteID
        <id>6</id>
      </interface_config>
      <motetype_identifier>mtype547873133</motetype_identifier>
    </mote>
    <mote>
      <interface_config>
        org.contikios.cooja.interfaces.Position
        <x>34.52946706220632</x>
        <y>71.18938377902246</y>
        <z>0.0</z>
      </interface_config>
      <interface_config>
        org.contikios.cooja.contikimote.interfaces.ContikiMoteID
        <id>7</id>
      </interface_config>
      <motetype_identifier>mtype547873133</motetype_identifier>
    </mote>
    <mote>
      <interface_config>
        org.contikios.cooja.interfaces.Position
        <x>68.96468705584522</x>
        <y>46.699412310903334</y>
        <z>0.0</z>
      </interface_config>
      <interface_config>
        org.contikios.cooja.contikimote.interfaces.ContikiMoteID
        <id>8</id>
      </interface_config>
      <motetype_identifier>mtype547873133</motetype_identifier>
    </mote>
    <mote>
      <interface_config>
        org.contikios.cooja.interfaces.Position
        <x>67.47288032116825</x>
        <y>28.464459635806065</y>
        <z>0.0</z>
      </interface_config>
      <interface_config>
        org.contikios.cooja.contikimote.interfaces.ContikiMoteID
        <id>9</id>
      </interface_config>
      <motetype_identifier>mtype547873133</motetype_identifier>
    </mote>
    <mote>
      <interface_config>
        org.contikios.cooja.interfaces.Position
        <x>69.91731238448307</x>
        <y>50.34229943693274</y>
        <z>0.0</z>
      </interface_config>
      <interface_config>
        org.contikios.cooja.contikimote.interfaces.ContikiMoteID
        <id>10</id>
      </interface_config>
      <motetype_identifier>mtype547873133</motetype_identifier>
    </mote>
    <mote>
      <interface_config>
        org.contikios.cooja.interfaces.Position
        <x>13.270328272847848</x>
        <y>13.833726204836857</y>
        <z>0.0</z>
      </interface_config>
      <interface_config>
        org.contikios.cooja.contikimote.interfaces.ContikiMoteID
        <id>11</id>
      </interface_config>
      <motetype_identifier>mtype547873133</motetype_identifier>
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
      <viewport>1.9138138544192345 0.0 0.0 1.9138138544192345 -23.141176545243322 4.316286130154822</viewport>
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
    <z>3</z>
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


var attacker = selectAttacker(); 

///////////////////////////////////////////////////////////////////////////////////////////

// Start Attack
GENERATE_MSG(26000000, "continue");
YIELD_THEN_WAIT_UNTIL(msg.equals("continue"));

log.log("Network DIS flooding attack from " + attacker.getID() + "!\n");
sim.getEventCentral().logEvent("attack", "dfa:" + attacker.getID());

/* Configure DIS flood random attack */
setBool(attacker, 'network_attacks_rpl_dfa', true);
setInt16(attacker, 'network_attacks_rpl_dfa_period', 1000);


// Stop Attack after 15 mins
GENERATE_MSG(900000, "stop-attack");
YIELD_THEN_WAIT_UNTIL(msg.equals("stop-attack"));

log.log("Stopped - Network DIS flooding attack from " + attacker.getID() + "!\n");
sim.getEventCentral().logEvent("stop-attack", "dfa:" + attacker.getID());

setBool(attacker, 'network_attacks_rpl_dfa', true);
setInt16(attacker, 'network_attacks_rpl_dfa_period', 10000);

///////////////////////////////////////////////////////////////////////////////////////////

// Start Attack after 15 minutes
GENERATE_MSG(900000, "continue");
YIELD_THEN_WAIT_UNTIL(msg.equals("continue"));

log.log("Network DIS flooding attack from " + attacker.getID() + "!\n");
sim.getEventCentral().logEvent("attack", "dfa:" + attacker.getID());

/* Configure DIS flood random attack */
setBool(attacker, 'network_attacks_rpl_dfa', true);
setInt16(attacker, 'network_attacks_rpl_dfa_period', 1000);


// Stop Attack after 15 mins
GENERATE_MSG(900000, "stop-attack");
YIELD_THEN_WAIT_UNTIL(msg.equals("stop-attack"));

log.log("Stopped - Network DIS flooding attack from " + attacker.getID() + "!\n");
sim.getEventCentral().logEvent("stop-attack", "dfa:" + attacker.getID());

setBool(attacker, 'network_attacks_rpl_dfa', true);
setInt16(attacker, 'network_attacks_rpl_dfa_period', 10000);

///////////////////////////////////////////////////////////////////////////////////////////

// Start Attack after 15 minutes
GENERATE_MSG(900000, "continue");
YIELD_THEN_WAIT_UNTIL(msg.equals("continue"));

log.log("Network DIS flooding attack from " + attacker.getID() + "!\n");
sim.getEventCentral().logEvent("attack", "dfa:" + attacker.getID());

/* Configure DIS flood random attack */
setBool(attacker, 'network_attacks_rpl_dfa', true);
setInt16(attacker, 'network_attacks_rpl_dfa_period', 1000);


// Stop Attack after 15 mins
GENERATE_MSG(900000, "stop-attack");
YIELD_THEN_WAIT_UNTIL(msg.equals("stop-attack"));

log.log("Stopped - Network DIS flooding attack from " + attacker.getID() + "!\n");
sim.getEventCentral().logEvent("stop-attack", "dfa:" + attacker.getID());

setBool(attacker, 'network_attacks_rpl_dfa', true);
setInt16(attacker, 'network_attacks_rpl_dfa_period', 10000);

///////////////////////////////////////////////////////////////////////////////////////////
// Start Attack after 15 minutes
GENERATE_MSG(900000, "continue");
YIELD_THEN_WAIT_UNTIL(msg.equals("continue"));

log.log("Network DIS flooding attack from " + attacker.getID() + "!\n");
sim.getEventCentral().logEvent("attack", "dfa:" + attacker.getID());

/* Configure DIS flood random attack */
setBool(attacker, 'network_attacks_rpl_dfa', true);
setInt16(attacker, 'network_attacks_rpl_dfa_period', 1000);


// Stop Attack after 15 mins
GENERATE_MSG(900000, "stop-attack");
YIELD_THEN_WAIT_UNTIL(msg.equals("stop-attack"));

log.log("Stopped - Network DIS flooding attack from " + attacker.getID() + "!\n");
sim.getEventCentral().logEvent("stop-attack", "dfa:" + attacker.getID());

setBool(attacker, 'network_attacks_rpl_dfa', true);
setInt16(attacker, 'network_attacks_rpl_dfa_period', 10000);

///////////////////////////////////////////////////////////////////////////////////////////
// Start Attack after 15 minutes
GENERATE_MSG(900000, "continue");
YIELD_THEN_WAIT_UNTIL(msg.equals("continue"));

log.log("Network DIS flooding attack from " + attacker.getID() + "!\n");
sim.getEventCentral().logEvent("attack", "dfa:" + attacker.getID());

/* Configure DIS flood random attack */
setBool(attacker, 'network_attacks_rpl_dfa', true);
setInt16(attacker, 'network_attacks_rpl_dfa_period', 1000);


// Stop Attack after 15 mins
GENERATE_MSG(900000, "stop-attack");
YIELD_THEN_WAIT_UNTIL(msg.equals("stop-attack"));

log.log("Stopped - Network DIS flooding attack from " + attacker.getID() + "!\n");
sim.getEventCentral().logEvent("stop-attack", "dfa:" + attacker.getID());

setBool(attacker, 'network_attacks_rpl_dfa', true);
setInt16(attacker, 'network_attacks_rpl_dfa_period', 10000);

///////////////////////////////////////////////////////////////////////////////////////////
// Start Attack after 15 minutes
GENERATE_MSG(900000, "continue");
YIELD_THEN_WAIT_UNTIL(msg.equals("continue"));

log.log("Network DIS flooding attack from " + attacker.getID() + "!\n");
sim.getEventCentral().logEvent("attack", "dfa:" + attacker.getID());

/* Configure DIS flood random attack */
setBool(attacker, 'network_attacks_rpl_dfa', true);
setInt16(attacker, 'network_attacks_rpl_dfa_period', 1000);


// Stop Attack after 15 mins
GENERATE_MSG(900000, "stop-attack");
YIELD_THEN_WAIT_UNTIL(msg.equals("stop-attack"));

log.log("Stopped - Network DIS flooding attack from " + attacker.getID() + "!\n");
sim.getEventCentral().logEvent("stop-attack", "dfa:" + attacker.getID());

setBool(attacker, 'network_attacks_rpl_dfa', true);
setInt16(attacker, 'network_attacks_rpl_dfa_period', 10000);

///////////////////////////////////////////////////////////////////////////////////////////
// Start Attack after 15 minutes
GENERATE_MSG(900000, "continue");
YIELD_THEN_WAIT_UNTIL(msg.equals("continue"));

log.log("Network DIS flooding attack from " + attacker.getID() + "!\n");
sim.getEventCentral().logEvent("attack", "dfa:" + attacker.getID());

/* Configure DIS flood random attack */
setBool(attacker, 'network_attacks_rpl_dfa', true);
setInt16(attacker, 'network_attacks_rpl_dfa_period', 1000);


// Stop Attack after 15 mins
GENERATE_MSG(900000, "stop-attack");
YIELD_THEN_WAIT_UNTIL(msg.equals("stop-attack"));

log.log("Stopped - Network DIS flooding attack from " + attacker.getID() + "!\n");
sim.getEventCentral().logEvent("stop-attack", "dfa:" + attacker.getID());

setBool(attacker, 'network_attacks_rpl_dfa', true);
setInt16(attacker, 'network_attacks_rpl_dfa_period', 10000);

///////////////////////////////////////////////////////////////////////////////////////////
// Start Attack after 15 minutes
GENERATE_MSG(900000, "continue");
YIELD_THEN_WAIT_UNTIL(msg.equals("continue"));

log.log("Network DIS flooding attack from " + attacker.getID() + "!\n");
sim.getEventCentral().logEvent("attack", "dfa:" + attacker.getID());

/* Configure DIS flood random attack */
setBool(attacker, 'network_attacks_rpl_dfa', true);
setInt16(attacker, 'network_attacks_rpl_dfa_period', 1000);


// Stop Attack after 15 mins
GENERATE_MSG(900000, "stop-attack");
YIELD_THEN_WAIT_UNTIL(msg.equals("stop-attack"));

log.log("Stopped - Network DIS flooding attack from " + attacker.getID() + "!\n");
sim.getEventCentral().logEvent("stop-attack", "dfa:" + attacker.getID());

setBool(attacker, 'network_attacks_rpl_dfa', true);
setInt16(attacker, 'network_attacks_rpl_dfa_period', 10000);

///////////////////////////////////////////////////////////////////////////////////////////
// Start Attack after 15 minutes
GENERATE_MSG(900000, "continue");
YIELD_THEN_WAIT_UNTIL(msg.equals("continue"));

log.log("Network DIS flooding attack from " + attacker.getID() + "!\n");
sim.getEventCentral().logEvent("attack", "dfa:" + attacker.getID());

/* Configure DIS flood random attack */
setBool(attacker, 'network_attacks_rpl_dfa', true);
setInt16(attacker, 'network_attacks_rpl_dfa_period', 1000);


// Stop Attack after 15 mins
GENERATE_MSG(900000, "stop-attack");
YIELD_THEN_WAIT_UNTIL(msg.equals("stop-attack"));

log.log("Stopped - Network DIS flooding attack from " + attacker.getID() + "!\n");
sim.getEventCentral().logEvent("stop-attack", "dfa:" + attacker.getID());

setBool(attacker, 'network_attacks_rpl_dfa', true);
setInt16(attacker, 'network_attacks_rpl_dfa_period', 10000);

///////////////////////////////////////////////////////////////////////////////////////////
// Start Attack after 15 minutes
GENERATE_MSG(900000, "continue");
YIELD_THEN_WAIT_UNTIL(msg.equals("continue"));

log.log("Network DIS flooding attack from " + attacker.getID() + "!\n");
sim.getEventCentral().logEvent("attack", "dfa:" + attacker.getID());

/* Configure DIS flood random attack */
setBool(attacker, 'network_attacks_rpl_dfa', true);
setInt16(attacker, 'network_attacks_rpl_dfa_period', 1000);


// Stop Attack after 15 mins
GENERATE_MSG(900000, "stop-attack");
YIELD_THEN_WAIT_UNTIL(msg.equals("stop-attack"));

log.log("Stopped - Network DIS flooding attack from " + attacker.getID() + "!\n");
sim.getEventCentral().logEvent("stop-attack", "dfa:" + attacker.getID());

setBool(attacker, 'network_attacks_rpl_dfa', true);
setInt16(attacker, 'network_attacks_rpl_dfa_period', 10000);

///////////////////////////////////////////////////////////////////////////////////////////
// Start Attack after 15 minutes
GENERATE_MSG(900000, "continue");
YIELD_THEN_WAIT_UNTIL(msg.equals("continue"));

log.log("Network DIS flooding attack from " + attacker.getID() + "!\n");
sim.getEventCentral().logEvent("attack", "dfa:" + attacker.getID());

/* Configure DIS flood random attack */
setBool(attacker, 'network_attacks_rpl_dfa', true);
setInt16(attacker, 'network_attacks_rpl_dfa_period', 1000);


// Stop Attack after 15 mins
GENERATE_MSG(900000, "stop-attack");
YIELD_THEN_WAIT_UNTIL(msg.equals("stop-attack"));

log.log("Stopped - Network DIS flooding attack from " + attacker.getID() + "!\n");
sim.getEventCentral().logEvent("stop-attack", "dfa:" + attacker.getID());

setBool(attacker, 'network_attacks_rpl_dfa', true);
setInt16(attacker, 'network_attacks_rpl_dfa_period', 10000);

///////////////////////////////////////////////////////////////////////////////////////////
// Start Attack after 15 minutes
GENERATE_MSG(900000, "continue");
YIELD_THEN_WAIT_UNTIL(msg.equals("continue"));

log.log("Network DIS flooding attack from " + attacker.getID() + "!\n");
sim.getEventCentral().logEvent("attack", "dfa:" + attacker.getID());

/* Configure DIS flood random attack */
setBool(attacker, 'network_attacks_rpl_dfa', true);
setInt16(attacker, 'network_attacks_rpl_dfa_period', 1000);


// Stop Attack after 15 mins
GENERATE_MSG(900000, "stop-attack");
YIELD_THEN_WAIT_UNTIL(msg.equals("stop-attack"));

log.log("Stopped - Network DIS flooding attack from " + attacker.getID() + "!\n");
sim.getEventCentral().logEvent("stop-attack", "dfa:" + attacker.getID());

setBool(attacker, 'network_attacks_rpl_dfa', true);
setInt16(attacker, 'network_attacks_rpl_dfa_period', 10000);

///////////////////////////////////////////////////////////////////////////////////////////
// Start Attack after 15 minutes
GENERATE_MSG(900000, "continue");
YIELD_THEN_WAIT_UNTIL(msg.equals("continue"));

log.log("Network DIS flooding attack from " + attacker.getID() + "!\n");
sim.getEventCentral().logEvent("attack", "dfa:" + attacker.getID());

/* Configure DIS flood random attack */
setBool(attacker, 'network_attacks_rpl_dfa', true);
setInt16(attacker, 'network_attacks_rpl_dfa_period', 1000);


// Stop Attack after 15 mins
GENERATE_MSG(900000, "stop-attack");
YIELD_THEN_WAIT_UNTIL(msg.equals("stop-attack"));

log.log("Stopped - Network DIS flooding attack from " + attacker.getID() + "!\n");
sim.getEventCentral().logEvent("stop-attack", "dfa:" + attacker.getID());

setBool(attacker, 'network_attacks_rpl_dfa', true);
setInt16(attacker, 'network_attacks_rpl_dfa_period', 10000);

///////////////////////////////////////////////////////////////////////////////////////////
// Start Attack after 15 minutes
GENERATE_MSG(900000, "continue");
YIELD_THEN_WAIT_UNTIL(msg.equals("continue"));

log.log("Network DIS flooding attack from " + attacker.getID() + "!\n");
sim.getEventCentral().logEvent("attack", "dfa:" + attacker.getID());

/* Configure DIS flood random attack */
setBool(attacker, 'network_attacks_rpl_dfa', true);
setInt16(attacker, 'network_attacks_rpl_dfa_period', 1000);


// Stop Attack after 15 mins
GENERATE_MSG(900000, "stop-attack");
YIELD_THEN_WAIT_UNTIL(msg.equals("stop-attack"));

log.log("Stopped - Network DIS flooding attack from " + attacker.getID() + "!\n");
sim.getEventCentral().logEvent("stop-attack", "dfa:" + attacker.getID());

setBool(attacker, 'network_attacks_rpl_dfa', true);
setInt16(attacker, 'network_attacks_rpl_dfa_period', 10000);

///////////////////////////////////////////////////////////////////////////////////////////
// Start Attack after 15 minutes
GENERATE_MSG(900000, "continue");
YIELD_THEN_WAIT_UNTIL(msg.equals("continue"));

log.log("Network DIS flooding attack from " + attacker.getID() + "!\n");
sim.getEventCentral().logEvent("attack", "dfa:" + attacker.getID());

/* Configure DIS flood random attack */
setBool(attacker, 'network_attacks_rpl_dfa', true);
setInt16(attacker, 'network_attacks_rpl_dfa_period', 1000);


// Stop Attack after 15 mins
GENERATE_MSG(900000, "stop-attack");
YIELD_THEN_WAIT_UNTIL(msg.equals("stop-attack"));

log.log("Stopped - Network DIS flooding attack from " + attacker.getID() + "!\n");
sim.getEventCentral().logEvent("stop-attack", "dfa:" + attacker.getID());

setBool(attacker, 'network_attacks_rpl_dfa', true);
setInt16(attacker, 'network_attacks_rpl_dfa_period', 10000);

///////////////////////////////////////////////////////////////////////////////////////////



//GENERATE_MSG(26000000, "continue");
//YIELD_THEN_WAIT_UNTIL(msg.equals("continue"));




log.testOK();</script>
      <active>true</active>
    </plugin_config>
    <width>600</width>
    <z>2</z>
    <height>642</height>
    <location_x>900</location_x>
    <location_y>0</location_y>
  </plugin>
</simconf>