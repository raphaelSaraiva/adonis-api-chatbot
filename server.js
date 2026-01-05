'use strict';

require('@adonisjs/ignitor');
require('dotenv').config();

const https = require('https');
const fs = require('fs');
const path = require('path');

const { Ignitor } = require('@adonisjs/ignitor');

const HOST = process.env.HOST || '0.0.0.0';
const PORT = Number(process.env.PORT || 3333);

// ✅ chaves "dentro da API" (pasta do projeto)
const keyPath = path.join(__dirname, 'ssl', 'server.key');
const certPath = path.join(__dirname, 'ssl', 'server.crt');

if (!fs.existsSync(keyPath) || !fs.existsSync(certPath)) {
  console.error('❌ Certificado não encontrado em:', { keyPath, certPath });
  process.exit(1);
}

const tlsOptions = {
  key: fs.readFileSync(keyPath),
  cert: fs.readFileSync(certPath),
};

new Ignitor(require('@adonisjs/fold'))
  .appRoot(__dirname)
  .fireHttpServer()
  .then((adonisServer) => {
    const server = https.createServer(tlsOptions, adonisServer.handle);

    server.listen(PORT, HOST, () => {
      console.log(`✅ HTTPS on https://${HOST}:${PORT}`);
    });
  })
  .catch((e) => {
    console.error(e);
    process.exit(1);
  });
