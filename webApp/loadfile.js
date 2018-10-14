function LoadFile() {

  var oFrame = document.getElementById("frmFile");
  var strRawContents = oFrame.contentWindow.document.body.childNodes[0].innerHTML;

  while (strRawContents.indexOf("\r") >= 0)
    strRawContents = strRawContents.replace("\r", "");

  var arrLines = strRawContents.split("\n");

  var text = document.getElementById("csvOutput");
  for (var i = 0; i < arrLines.length; i++) {
    var curLine = arrLines[i];
    // var frame = document.getElementById('csvOutput').innherHTML(curLine);

  }
}
