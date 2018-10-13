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

// load the soundcloud
setupAudioNodes();
loadSound("sound_track.mp3");

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
