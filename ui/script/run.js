// // basic JavaScript function
// function runCodeFromTerminal() {
//     let string1 = "welcome ";
//     let string2 = "to the";
//     let string3 = "tutorialspoint!"
//     console.log( string1, string2, string3 );
//  }
 
 
//  // call the function on run the file.
// //  runCodeFromTerminal();


// const { exec } = require('node:child_process');

// // exec('"/path/to/test file/test.sh" arg1 arg2');
// // // Double quotes are used so that the space in the path is not interpreted as
// // // a delimiter of multiple arguments.

// // exec('echo "The \\$HOME variable is $HOME"');
// // // The $HOME variable is escaped in the first instance, but not in the second.

// exec('"cmd.sh"');

// temperature-listener.js

const { spawn } = require('child_process');
const temperatures = []; // Store readings

// const sensor = spawn('python38', ['../python/app.py']);
const sensor = spawn('python38', ['../../morpher/test/test_morpher.py']);
sensor.stdout.on('data', function(data) {

    // convert Buffer object to Float
    // temperatures.push(parseFloat(data));
    temperatures.push(String(data));
    console.log(temperatures);
});