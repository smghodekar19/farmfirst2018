const bodyParser = require("body-parser");
const express = require("express");
const app = express();
var path = require('path');
var fs = require('fs');
var http = require('http');
var createFile = require('create-file');
var PythonShell = require('python-shell');
var jsonfile = require('jsonfile')



var urlencoded = bodyParser.urlencoded({ extended: false });

//var coun=0;
app.post('/abc',urlencoded, (req, res) => {
    var location = req.body.location;
    console.log('location: '+location);
    //res.send('You are now at ' + location);
   


 //var fileName = './file/user'+coun+'.json';
//createFile(fileName, 'location: '+location+" ", function (err) {
  //  console.log("file is created : ",fileName);
   // coun++;
 //});
 
//var file = './file/user'+coun+'.json';
  //jsonfile.writeFile(file, '{ location : '+location+ '}', function (err) {
 // console.error(err)
 // coun++;
//})
  
//python code here *************************
var options = {
    mode: 'text',
   // pythonPath: 'C:/Python27',
    pythonOptions: ['-u'],
    scriptPath: 'code',
    args: location
  };


 PythonShell.run('mlrnoclass.py',options, function (err, results) {
   if (err) throw err;
    // results is an array consisting of messages collected during execution
    console.log('results: %j', results[0]);
    res.send( results[0]);


    //var jsonObj = JSON.parse(results);
   // console.log(jsonObj);
    
    // stringify JSON Object
   // var jsonContent = JSON.stringify(jsonObj);
   // console.log(jsonContent);
    /*
    fs.writeFile("output.json", jsonContent, 'utf8', function (err) {
        if (err) {
            console.log("An error occured while writing JSON Object to File.");
            return console.log(err);
        }
    
       */ //console.log("JSON file has been saved.");
      // res.json( results);
     
 });

    /* no need code
    var file = './file/user'+coun+'.json';
    jsonfile.readFile(file, function(err, obj) {  
    console.dir(jsonfile.readFileSync(file))
    res.writeHead(200,{'Content-Type' : 'application/json'});
    res.end(JSON.stringify(obj));
    })

    
    */

  //}); python ends here *******************************

 
 }); 

app.use('/', (req, res) => {
 // res.sendFile(`${__dirname}/demo.html`);
//});
if(req.url === "/" || req.url === "/home") {

    fs.readFile("./public/index.html", "UTF-8", function(err,html){
        res.writeHead(200, {'Content-Type': 'text/html'});
        res.end(html);
    })
}else if(req.url.match("\.css$")){
    
    var csspath = path.join(__dirname,'public',req.url);
    var fileStream = fs.createReadStream(csspath, "UTf-8");
    res.writeHead(200,{"content-Type": "text/css"});
    fileStream.pipe(res);

}
else if(req.url.match("\.js$")){
    
    var jspath = path.join(__dirname,'public',req.url);
    var fileStream = fs.createReadStream(jspath, "UTf-8");
    res.writeHead(200,{"content-Type": "scripts/js"});
    fileStream.pipe(res);

}
else if(req.url.match("\.png$")){
    
    var pngpath = path.join(__dirname,'public',req.url);
    var fileStream = fs.createReadStream(pngpath);
    res.writeHead(200,{"content-Type": "images/png"});
    fileStream.pipe(res);  
}
else if(req.url.match("\.jpg$")){
    
    var jpgpath = path.join(__dirname,'public',req.url);
    var fileStream = fs.createReadStream(jpgpath);
    res.writeHead(200,{"content-Type": "images/jpg"});
    fileStream.pipe(res);  
}
else if(req.url === '/about'){
   res.writeHead(200,{'content-Type': 'text/html'});
   fs.createReadStream(__dirname + '/public/aboutus.html').pipe(res); 
  
}

else if(req.url === '/contact'){
    res.writeHead(200,{'content-Type': 'text/html'});
    fs.createReadStream(__dirname + '/public/contact.html').pipe(res); 
     

 }
 //else if(req.url === '/result'){
    
 app.get('/res',function(req, res,next){
     var demo = {
        title: "demo",
        location:location
     };
 res.render('../views/res',demo)
 })

 //}
console.log(req.url);
});
 /*  routing from here    */




app.listen(8900, () => {
  console.log("Started on http://localhost:8900");
});

