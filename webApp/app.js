"use strict";

// Create an instance
var wavesurfer;
setTimeout(function() {
  window.location.reload(1);
}, 4000);
// Init & load
document.addEventListener("DOMContentLoaded", function() {
  var micBtn = document.querySelector("#micBtn");

  // Init wavesurfer
  wavesurfer = WaveSurfer.create({
    container: "#waveform",
    waveColor: "#f7f7f7",
    progressColor: "#4f4f4f",
    barHeight: 5,
    barWidth: 0.1,
    barGap: -1,
    interact: false,
    cursorWidth: 0,
    normalize: true,
    plugins: [WaveSurfer.microphone.create()]
  });

  wavesurfer.microphone.on("deviceReady", function() {
    console.info("Device ready!");
  });
  wavesurfer.microphone.on("deviceError", function(code) {
    console.warn("Device error: " + code);
  });

  // start/stop mic on button click
  window.onload = function() {
    if (wavesurfer.microphone.active) {
      wavesurfer.microphone.stop();
    } else {
      wavesurfer.microphone.start();
    }

    // start running python scripts
  };
});
