# Documentação da API de Pagamentos

Esta documentação descreve os endpoints disponíveis na API de pagamentos interna, usada por times de produto para processar cobranças, estornos e consultar status de transações.

## Autenticação

Todas as requisições devem incluir um header `Authorization: Bearer <token>`. Os tokens são emitidos pelo serviço de identidade interno e expiram após 24 horas. Requisições sem token válido retornam `401 Unauthorized`.

Para obter um token, envie uma requisição POST para `/auth/token` com as credenciais do serviço. O corpo da resposta contém o token e o timestamp de expiração.

## Criando uma cobrança

O endpoint `POST /charges` cria uma nova cobrança. Os campos obrigatórios são `amount` (em centavos), `currency` (código ISO 4217) e `customer_id`. Campos opcionais incluem `description` e `metadata`.

Se a cobrança for aprovada, a API retorna `201 Created` com o objeto da cobrança, incluindo o campo `status` igual a `succeeded`. Se for recusada pelo processador de pagamento, retorna `402 Payment Required` com o motivo da recusa em `decline_code`.

Cobranças duplicadas (mesmo `idempotency_key` dentro de 24 horas) retornam o resultado da primeira tentativa sem processar novamente.

## Estornos

O endpoint `POST /refunds` estorna uma cobrança existente, total ou parcialmente. É necessário informar `charge_id`. Se `amount` não for informado, o estorno é integral.

Estornos parciais podem ser feitos múltiplas vezes até atingir o valor total da cobrança original. Após atingir o limite, novas tentativas retornam `400 Bad Request`.

## Consultando status de transações

O endpoint `GET /charges/{id}` retorna o estado atual de uma cobrança. Os possíveis valores de `status` são: `pending`, `succeeded`, `failed` e `refunded`.

Para consultas em lote, use `GET /charges?customer_id={id}&limit={n}`, que retorna uma lista paginada. O parâmetro `limit` aceita no máximo 100 registros por página.

## Rate limiting

A API aplica rate limiting de 100 requisições por minuto por token. Ao exceder o limite, a resposta é `429 Too Many Requests` com um header `Retry-After` indicando quantos segundos aguardar antes de tentar novamente.

## Webhooks

É possível registrar um endpoint de webhook para receber eventos assíncronos como `charge.succeeded`, `charge.failed` e `refund.created`. Cada evento é assinado com HMAC-SHA256 usando o secret configurado; o header `X-Signature` deve ser validado antes de processar o payload.

Eventos não confirmados (sem resposta `200`) são reenviados com backoff exponencial por até 72 horas, após o que são descartados.
