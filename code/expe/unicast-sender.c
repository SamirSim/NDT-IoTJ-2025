/*
 * Copyright (c) 2011, Swedish Institute of Computer Science.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. Neither the name of the Institute nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE INSTITUTE AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE INSTITUTE OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 *
 * This file is part of the Contiki operating system.
 *
 */

#include "contiki.h"
#include "lib/random.h"
#include "sys/ctimer.h"
#include "sys/etimer.h"
#include "net/ip/uip.h"
#include "net/ipv6/uip-ds6.h"
#include "net/ip/uip-debug.h"

#include "sys/node-id.h"

#include "simple-udp.h"
#include "servreg-hack.h"

#include <stdio.h>
#include <string.h>

#define UDP_PORT 1234
#define SERVICE_ID 190

unsigned short cpt = 0;
uint32_t cpt2 = 0;

#define SEND_INTERVAL		(1 * CLOCK_SECOND)
//#define SEND_TIME		(random_rand() % (SEND_INTERVAL))

#define CONFIG_INTERVAL		(100 * CLOCK_SECOND)

// Define the valid range for CSMA_MIN_BE
#define MIN_MIN_BE     0   // Minimum value for min_be_value (default: 3)
#define MAX_MIN_BE     7   // Maximum value for min_be_value (default: 8)

// Define the valid range for CSMA_MAX_BE
#define MIN_MAX_BE     3   // Minimum value for max_be_value (default: 4)
#define MAX_MAX_BE     8  // Maximum value for max_be_value (default: 10)

// Define the valid range for CSMA_MAX_BACKOFF
#define MIN_MAX_BACKOFF  0   // Minimum value for max_backoff (default: 1)
#define MAX_MAX_BACKOFF  5   // Maximum value for max_backoff (default: 8)

// Define the valid range for CSMA_FRAME_RETRIES
#define MIN_FRAME_RETRIES  0  // Minimum value for frame_retries (default: 1)
#define MAX_FRAME_RETRIES  7  // Maximum value for frame_retries (default: 7)

static struct simple_udp_connection unicast_connection;

/*---------------------------------------------------------------------------*/
PROCESS(unicast_sender_process, "Unicast sender example process");
AUTOSTART_PROCESSES(&unicast_sender_process);
/*---------------------------------------------------------------------------*/
static void
receiver(struct simple_udp_connection *c,
         const uip_ipaddr_t *sender_addr,
         uint16_t sender_port,
         const uip_ipaddr_t *receiver_addr,
         uint16_t receiver_port,
         const uint8_t *data,
         uint16_t datalen)
{
  printf("Data received on port %d from port %d with length %d\n",
         receiver_port, sender_port, datalen);
}
/*---------------------------------------------------------------------------*/
static void
set_global_address(void)
{
  uip_ipaddr_t ipaddr;
  int i;
  uint8_t state;

  uip_ip6addr(&ipaddr, UIP_DS6_DEFAULT_PREFIX, 0, 0, 0, 0, 0, 0, 0);
  uip_ds6_set_addr_iid(&ipaddr, &uip_lladdr);
  uip_ds6_addr_add(&ipaddr, 0, ADDR_AUTOCONF);

  printf("IPv6 addresses: ");
  for(i = 0; i < UIP_DS6_ADDR_NB; i++) {
    state = uip_ds6_if.addr_list[i].state;
    if(uip_ds6_if.addr_list[i].isused &&
       (state == ADDR_TENTATIVE || state == ADDR_PREFERRED)) {
      uip_debug_ipaddr_print(&uip_ds6_if.addr_list[i].ipaddr);
      printf("\n");
    }
  }
}
/*---------------------------------------------------------------------------*/
PROCESS_THREAD(unicast_sender_process, ev, data)
{
  static struct etimer periodic_timer, send_timer, config_timer;
  uip_ipaddr_t *addr;
  uint32_t id;

  PROCESS_BEGIN();

  servreg_hack_init();

  set_global_address();

  simple_udp_register(&unicast_connection, UDP_PORT,
                      NULL, UDP_PORT, receiver);

  // Set initial timers: one for 5 seconds, the other for 10 seconds
  etimer_set(&config_timer, CONFIG_INTERVAL); // Config change every 10 seconds
  etimer_set(&send_timer, SEND_INTERVAL); // Config change every 10 seconds

  while(1) {
    PROCESS_WAIT_EVENT();

    //id = 0;

    if (etimer_expired(&send_timer)) { 
      //const linkaddr_t *mac_address = &linkaddr_node_addr;
      //uint32_t randnum = (mac_address->u8[6] << 8) | mac_address->u8[7];
      //random_init(randnum);
      //id = random_rand() + randnum;
      //printf("Random number: %ld %ld \n", randnum, id);
      addr = servreg_hack_lookup(SERVICE_ID);
      if(addr != NULL) {
        char buf[20];
        //uip_debug_ipaddr_print(addr);
        //printf("\n");
        snprintf(buf, sizeof(buf), "%07ld", cpt2); 
        cpt2 = cpt2 + 1;
        printf("Sending packet content: %s\n", buf);
        simple_udp_sendto(&unicast_connection, buf, strlen(buf) + 1, addr);
      } else {
        printf("Service %d not found\n", SERVICE_ID);
      }

      etimer_reset(&send_timer);
    }

    if (etimer_expired(&config_timer)) {
      // The following lines are used to generate the same random configuration for all the nodes
      //random_init(cpt);
      //cpt = cpt + 1;

      const linkaddr_t *mac_address = &linkaddr_node_addr;
      uint32_t randnum = (mac_address->u8[6] << 8) | mac_address->u8[7];
      
      random_init(randnum+cpt);
      cpt = cpt + 1;

      //printf("cpt and random_rand() %d %d\n", cpt, random_rand());
      // Randomly generate values within the valid ranges for each parameter
      uint8_t min_be_value = (random_rand() % (MAX_MIN_BE - MIN_MIN_BE + 1)) + MIN_MIN_BE; // e.g., MIN_MIN_BE = 3, MAX_MIN_BE = 8
      uint8_t max_be_value = (random_rand() % (MAX_MAX_BE - min_be_value + 1)) + min_be_value; // e.g., MIN_MAX_BE = 4, MAX_MAX_BE = 10
      uint8_t max_backoff = (random_rand() % (MAX_MAX_BACKOFF - MIN_MAX_BACKOFF + 1)) + MIN_MAX_BACKOFF; // e.g., MIN_MAX_BACKOFF = 1, MAX_MAX_BACKOFF = 8
      uint8_t frame_retries = (random_rand() % (MAX_FRAME_RETRIES - MIN_FRAME_RETRIES + 1)) + MIN_FRAME_RETRIES; // e.g., MIN_FRAME_RETRIES = 1, MAX_FRAME_RETRIES = 7

      // Set the new values using the setters you implemented
      set_csma_min_be(min_be_value);
      set_csma_max_be(max_be_value);
      set_csma_max_backoff(max_backoff);
      set_csma_max_frame_retries(frame_retries);

      // Log the new configuration
      printf("The configuration has been changed to: CSMA_MIN_BE=%u, CSMA_MAX_BE=%u, CSMA_MAX_BACKOFF=%u, FRAME_RETRIES=%u\n",
            min_be_value, max_be_value, max_backoff, frame_retries);
      
      // Reset the timer for the next change (5 minutes = 300 seconds)
      etimer_reset(&config_timer);
    }
  }

  PROCESS_END();
}
/*---------------------------------------------------------------------------*/
