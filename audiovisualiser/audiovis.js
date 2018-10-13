if (!window.AudioContext) {
  if (!window.webkitAudioContext) {
    alert('no AudioContext found');
  }
  window.AudioContext = window.webkitAudioContext;
}

// create the audio context
var context = new webkitAudioContext();
var audioBuffer;
var sourceNode;

// setup an analyser
 analyser = context.createAnalyser();
 analyser.smoothingTimeConstant = 0.3;
 analyser.fftSize = 1024;

// setup a javascript node
  javascriptNode = context.createScriptProcessor(2048, 1, 1);

// load the soundcloud
setupAudioNodes();
loadSound("wagner-short.ogg");

function setupAudioNodes() {
    // create a buffer source source node,
    // and connect to a destination
    sourceNode = context.createBufferSource();
    sourceNide.connect(context.destination);
}

// load the specified sound
function loadSound(url) {
  var request = new XMLHttpRequest();
  request.open('GET', url, true);
  request.responseType = 'arraybuffer';

  // When loaded decode the data
  request.onload = function() {

      // decode the data
      context.decodeAudioData(request.response, function(buffer) {
        // when the audio is decoded play the sound
        playSound(buffer);
      }, onError);
  }
  request.send();
}


function playSound(buffer) {
   sourceNode.buffer = buffer;
   sourceNode.noteOn(0);
}

// log if an error occurs
function onError(e) {
  console.log(e);
}

// when the javascript node is called
// we use information from the analyzer node
// to draw the volume
javascriptNode.onaudioprocess = function() {

  // get the average for the first channel
  var array =  new Uint8Array(analyser.frequencyBinCount);
  analyser.getByteFrequencyData(array);
  var average = getAverageVolume(array);

  // get the average for the second channel
  var array2 =  new Uint8Array(analyser2.frequencyBinCount);
  analyser2.getByteFrequencyData(array2);
  var average2 = getAverageVolume(array2);

  // clear the current state
  ctx.clearRect(0, 0, 60, 130);

  // set the fill style
  ctx.fillStyle=gradient;

  // create the meters
  ctx.fillRect(0,130-average,25,130);
  ctx.fillRect(30,130-average2,25,130);
}

function getAverageVolume(array) {
  var values = 0;
  var average;

  var length = array.length;

  // get all the frequency amplitudes
  for (var i = 0; i < length; i++) {
    values += array[i];
  }

  average = values / length;
  return average;
}

function setupAudioNodes() {

  // setup a javascript node
  javascriptNode = context.createScriptProcessor(2048, 1, 1);
  // connect to destination, else it isn't called
  javascriptNode.connect(context.destination);

  // setup a analyzer
  analyser = context.createAnalyser();
  analyser.smoothingTimeConstant = 0.3;
  analyser.fftSize = 1024;

  analyser2 = context.createAnalyser();
  analyser2.smoothingTimeConstant = 0.0;
  analyser2.fftSize = 1024;

  // create a buffer source node
  sourceNode = context.createBufferSource();
  splitter = context.createChannelSplitter();

  // connect the source to the analyser and the splitter
  sourceNode.connect(splitter);

  // connect one of the outputs from the splitter to
  // the analyser
  splitter.connect(analyser,0,0);
  splitter.connect(analyser2,1,0);

  // we use the javascript node to draw at a
  // specific interval.
  analyser.connect(javascriptNode);

  // and connect to destination
  sourceNode.connect(context.destination);
}
