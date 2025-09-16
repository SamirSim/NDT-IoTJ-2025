#!/bin/bash
# exp.sh: launch experiment on IoT-lab, log & retrieve results from server

set -e

#---------------------- TEST ARGUMENTS ----------------------#
if [ "$#" -ne 7 ]; then
    echo "Usage: $0 <exp name> <interval (s)> <payload size (B)> <exp duration (m)> <packets per second> <list of transmitter nodes> <receiver node>"
    exit
fi
#---------------------- TEST ARGUMENTS ----------------------#

#--------------------- DEFINE VARIABLES ---------------------#
LOGIN="ssimoham"
SITE="grenoble"
IOTLAB="$LOGIN@$SITE.iot-lab.info"
CODEDIR="${HOME}/Desktop/LinkPrediction/iot-lab/parts/contiki/examples/ipv6/simple-udp-rpl"
EXPDIR="${HOME}/Desktop/LinkPrediction/traces-master"
FIRMWARE_TRANSMITTER="unicast-sender"
FIRMWARE_RECEIVER="unicast-receiver"
SENDER_NODES=$6
RECEIVER_NODE=$7
#--------------------- DEFINE VARIABLES ---------------------#

#----------------------- CATCH SIGINT -----------------------#
trap ctrl_c INT
function ctrl_c() {
    echo "Terminating experiment."
    iotlab-experiment stop -i "$EXPID"
    exit 1
}
#----------------------- CATCH SIGINT -----------------------#

#-------------------- CONFIGURE FIRMWARE --------------------#
sed -i "s/#define\ SEND_INTERVAL_SECONDS\ .*/#define\ SEND_INTERVAL_SECONDS\ $2/g" $CODEDIR/$FIRMWARE_TRANSMITTER.c
sed -i "s/#define\ SEND_BUFFER_SIZE\ .*/#define\ SEND_BUFFER_SIZE\ $3/g" $CODEDIR/$FIRMWARE_TRANSMITTER.c
sed -i "s/#define\ NB_PACKETS\ .*/#define\ NB_PACKETS\ $5/g" $CODEDIR/$FIRMWARE_TRANSMITTER.c
#-------------------- CONFIGURE FIRMWARE --------------------#

#--------------------- COMPILE FIRMWARE ---------------------#
cd $CODEDIR
make TARGET=iotlab-m3 -j8 || { echo "Compilation failed."; exit 1; }
#--------------------- COMPILE FIRMWARE ---------------------#

#-------------------- LAUNCH EXPERIMENTS --------------------#
cd $EXPDIR/scripts

# Construct the node list for submission
NODE_LIST="-l ${SITE},m3,${RECEIVER_NODE},${CODEDIR}/${FIRMWARE_RECEIVER}.iotlab-m3"
IFS=',' read -r -a SENDERS <<< "$SENDER_NODES"
for SENDER in "${SENDERS[@]}"; do
    NODE_LIST="$NODE_LIST -l ${SITE},m3,${SENDER},${CODEDIR}/${FIRMWARE_TRANSMITTER}.iotlab-m3"
done

# Submit the experiment and retrieve its ID
EXPID=$(iotlab-experiment submit -n $1 -d $4 $NODE_LIST | grep id | cut -d' ' -f6)

# Wait for the experiment to begin
iotlab-experiment wait -i $EXPID

# Wait for Contiki
sleep 10

# Run a script for logging and seeding
iotlab-experiment script -i $EXPID --run $SITE,script=serial_script.sh

# Wait for experiment termination
iotlab-experiment wait -i $EXPID --state Terminated
#-------------------- LAUNCH EXPERIMENTS --------------------#

#----------------------- RETRIEVE LOG -----------------------#
ssh $IOTLAB "tar -C ~/.iot-lab/${EXPID}/ -cvzf $1.tar.gz serial_output"
mkdir -p $EXPDIR/log/$EXPID
scp "$IOTLAB":~/$1.tar.gz $EXPDIR/log/$EXPID/$1.tar.gz
cd $EXPDIR/log/$EXPID/
tar -xvf $1.tar.gz 
#----------------------- RETRIEVE LOG -----------------------#

exit 0